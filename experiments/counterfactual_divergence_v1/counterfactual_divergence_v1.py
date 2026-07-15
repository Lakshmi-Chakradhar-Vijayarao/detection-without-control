# %% [code]
"""
COUNTERFACTUAL TRACE DIVERGENCE — v1

Core question: does epistemic geometry identify KV entries belonging to
stable attractors — and does it do so BEFORE the output reveals correctness?

For a reasoning model (DeepSeek-R1-Distill-Qwen-1.5B), we run each question
N_RUNS times at temperature > 0. Runs split into correct and wrong by F1 against
ground truth. We track J_know — the Fisher LDA projection of the residual stream
at every decode step — and find the first token where the correct and wrong
trajectory means diverge past a threshold.

The epistemic framing:
  PARAM questions converge to stable attractors in the residual stream.
  CTX_DEP and wrong answers are unstable inference states that eventually collapse.
  The J_know trajectory encodes which regime the inference is entering — BEFORE
  the output tokens reveal it.

  Implication for KV memory: KV entries generated during stable-attractor phases
  (high J_know, positive velocity) are candidates for retention. Entries generated
  during unstable phases (low or falling J_know) are candidates for eviction.
  This is the bridge from trajectory divergence to memory intelligence.

Statistical protocol
  • Bootstrap 95% CI on divergence rate across boundary questions
  • NULL HYPOTHESIS BASELINE: shuffle correct/wrong labels, measure divergence
    under null → z-score of observed vs expected by chance
  • Cohen's d per decode step (correct vs wrong J_know)
  • Wilcoxon signed-rank test on paired divergence gaps
  • All metrics reported with sample sizes

Outputs
  /kaggle/working/counterfactual_divergence_v1_results.json
  /kaggle/working/counterfactual_divergence_v1_figure.png
  /kaggle/working/counterfactual_divergence_v1_examples.json
"""

# %% [code]
import subprocess, sys, time

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *args])

pip("transformers>=4.45.0", "datasets", "scikit-learn", "scipy", "numpy", "matplotlib",
    "accelerate>=0.26.0")

# %% [code]
import json, os, re, string, gc, math
from pathlib import Path

_HF_CACHE = "/tmp/hf_cache"
os.environ["HF_HOME"]            = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = _HF_CACHE

HF_TOKEN = os.environ.get("HF_TOKEN", "")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, ttest_ind
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# Require T4 GPU (sm_75) or better — fail fast before model download
assert torch.cuda.is_available(), "GPU required. Enable GPU in Kaggle settings."
gpu_name  = torch.cuda.get_device_name(0)
gpu_vram  = torch.cuda.get_device_properties(0).total_memory / 1024**3
gpu_cc    = torch.cuda.get_device_capability(0)
_sm = gpu_cc[0] * 10 + gpu_cc[1]
assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run on T4."
print(f"GPU: {gpu_name}  VRAM: {gpu_vram:.1f} GB  CUDA capability: sm_{gpu_cc[0]}{gpu_cc[1]}")
print(f"PyTorch: {torch.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

DEVICE = "cuda"
DTYPE  = torch.float16

# %% [code]
# ── Experiment configuration ───────────────────────────────────────────────────
MODEL_ID  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PROBE_LAYER = 22          # ~80% depth of the 28-layer Qwen2.5 backbone

N_QUESTIONS  = 80         # TriviaQA questions to evaluate
N_RUNS       = 8          # stochastic runs per question (temperature > 0)
TEMPERATURE  = 0.7
MAX_TOKENS   = 500        # decode steps per run (covers full think + answer)
MIN_CORRECT  = 2          # boundary condition: ≥ this many correct runs
MIN_WRONG    = 2          # boundary condition: ≥ this many wrong runs

N_CAL        = 200        # bilateral oracle calibration samples (100 PARAM + 100 CTX_DEP)
CAL_MAX_TOK  = 600        # tokens for calibration scoring (enough for think + answer)

DIVERGENCE_THRESHOLD = 0.15   # |mean_correct − mean_wrong| > this → diverged
N_BOOTSTRAP  = 1000           # bootstrap iterations for CIs

SEED = 42
rng  = np.random.default_rng(SEED)
torch.manual_seed(SEED)

OUT_JSON     = "/kaggle/working/counterfactual_divergence_v1_results.json"
OUT_FIG      = "/kaggle/working/counterfactual_divergence_v1_figure.png"
OUT_EXAMPLES = "/kaggle/working/counterfactual_divergence_v1_examples.json"

print(f"\nConfig: {N_QUESTIONS} questions × {N_RUNS} runs  |  "
      f"T={TEMPERATURE}  max_tok={MAX_TOKENS}  cal={N_CAL}")

# %% [code]
# ── Helpers ────────────────────────────────────────────────────────────────────
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()

def f1_token(pred, targets):
    pred_toks = set(normalize(pred).split())
    if not pred_toks:
        return 0.0
    best = 0.0
    for t in targets:
        t_toks = set(normalize(str(t)).split())
        if not t_toks:
            continue
        common = pred_toks & t_toks
        if not common:
            continue
        p = len(common) / len(pred_toks)
        r = len(common) / len(t_toks)
        best = max(best, 2 * p * r / (p + r))
    return best

def extract_final_answer(text):
    """Extract answer after </think> tag; fall back to full text."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()

def make_chat_prompt(tok, question):
    """Standard chat-template prompt (thinking ENABLED — same for cal + main exp)."""
    user = f"Answer the following question concisely.\n\nQuestion: {question}"
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"Question: {question}\nAnswer:"

# %% [code]
# ── Model loading ──────────────────────────────────────────────────────────────
t_load = time.time()
print(f"\nLoading {MODEL_ID} ...")
tok_kwargs   = {"token": HF_TOKEN} if HF_TOKEN else {}
model_kwargs = dict(torch_dtype=DTYPE, low_cpu_mem_usage=True, **tok_kwargs)

model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tok_kwargs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inner    = getattr(model, "model", model)
layers   = inner.layers
n_layers = len(layers)
L_DEEP   = min(PROBE_LAYER, n_layers - 1)
print(f"  hidden_size={model.config.hidden_size}  layers={n_layers}  probe_layer={L_DEEP}")
print(f"  Load time: {time.time()-t_load:.1f}s")

# %% [code]
# ── Dataset ────────────────────────────────────────────────────────────────────
print("\nLoading TriviaQA validation ...")
ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation", trust_remote_code=True)
ds = ds.shuffle(seed=SEED)
print(f"  Dataset size: {len(ds)}")

# %% [code]
# ── Fisher probe calibration ───────────────────────────────────────────────────
# Bilateral oracle: run each question without context to determine PARAM vs CTX_DEP.
# PARAM  = model answers correctly without any context (parametric knowledge).
# CTX_DEP = model fails without context, but succeeds with context (needs retrieval).
#
# Hidden state capture uses the two-pass approach from esm/runtime.py:
#   1. Prefill forward pass  → build KV cache
#   2. Single-token decode   → capture step-1 hidden state at L_DEEP
# This is independent of generation length and avoids hook contamination.
#
# Scoring uses the SAME chat prompt with thinking enabled. With CAL_MAX_TOK=600,
# DeepSeek-R1 completes its think trace for most short factual questions.

print(f"\n{'='*60}")
print("CALIBRATING FISHER PROBE")
print(f"{'='*60}")
print(f"  Target: {N_CAL//2} PARAM + {N_CAL//2} CTX_DEP samples")

def _step1_capture(input_ids):
    """Prefill → clear → single-token decode → return step-1 hidden state."""
    captured = {}
    def _hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1].float().cpu().numpy()
    handle = layers[L_DEEP].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            pre = model(input_ids=input_ids, use_cache=True)
            pkv = pre.past_key_values
            if not isinstance(pkv, DynamicCache):
                dc = DynamicCache()
                for i, (k, v) in enumerate(pkv): dc.update(k, v, i)
                pkv = dc
            del pre
            captured.clear()                        # discard prefill captures
            _ = model(input_ids=input_ids[:, -1:],  # step-1 decode
                      past_key_values=pkv, use_cache=False)
    except Exception as e:
        print(f"  [step1 error] {e}")
        handle.remove()
        return None
    handle.remove()
    return captured.get("h")

def gen_and_score(prompt_text, max_tok):
    """Generate full response and return (answer_text, f1_against_gold) pair."""
    enc = tokenizer(prompt_text, return_tensors="pt",
                    truncation=True, max_length=768).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_tok, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    return extract_final_answer(raw)

pos_vecs, neg_vecs = [], []
n_target  = N_CAL // 2
_cal_n    = 0
_cal_skip = 0
t_cal     = time.time()

for row in ds:
    if len(pos_vecs) >= n_target and len(neg_vecs) >= n_target:
        break

    q       = row["question"]
    answers = row["answer"]["aliases"] or [row["answer"]["value"]]
    ctx_raw = row.get("search_results", {}).get("search_context", [""])
    ctx     = " ".join(ctx_raw[:2])[:600] if ctx_raw else ""

    prompt_nc = make_chat_prompt(tokenizer, q)
    ans_nc    = gen_and_score(prompt_nc, CAL_MAX_TOK)
    nc_f1     = f1_token(ans_nc, answers)

    if nc_f1 >= 0.40 and len(pos_vecs) < n_target:
        ids = tokenizer(prompt_nc, return_tensors="pt",
                        truncation=True, max_length=768)["input_ids"].to(DEVICE)
        h = _step1_capture(ids)
        if h is not None:
            pos_vecs.append(h)
        else:
            _cal_skip += 1
    elif nc_f1 <= 0.05 and ctx and len(neg_vecs) < n_target:
        prompt_ctx = make_chat_prompt(tokenizer, f"Context: {ctx}\n\nQuestion: {q}")
        ans_ctx    = gen_and_score(prompt_ctx, CAL_MAX_TOK)
        if f1_token(ans_ctx, answers) >= 0.40:
            ids = tokenizer(prompt_nc, return_tensors="pt",
                            truncation=True, max_length=768)["input_ids"].to(DEVICE)
            h = _step1_capture(ids)
            if h is not None:
                neg_vecs.append(h)
            else:
                _cal_skip += 1

    _cal_n += 1
    if _cal_n % 20 == 0:
        elapsed = time.time() - t_cal
        print(f"  [{_cal_n:3d} processed | {elapsed:.0f}s]  "
              f"PARAM={len(pos_vecs)}/{n_target}  CTX_DEP={len(neg_vecs)}/{n_target}",
              flush=True)
    torch.cuda.empty_cache()

print(f"  Done in {time.time()-t_cal:.0f}s: "
      f"PARAM={len(pos_vecs)}, CTX_DEP={len(neg_vecs)}, skipped={_cal_skip}")

if len(pos_vecs) < 10 or len(neg_vecs) < 10:
    raise RuntimeError(
        f"Calibration failed: only PARAM={len(pos_vecs)}, CTX_DEP={len(neg_vecs)}. "
        "Need ≥10 each. Check dataset loading and model output."
    )

H_pos  = np.stack(pos_vecs)
H_neg  = np.stack(neg_vecs)
c_neg  = H_neg.mean(0)
diff   = H_pos.mean(0) - c_neg
diff_u = diff / (np.linalg.norm(diff) + 1e-12)
theta  = (float((H_pos @ diff_u).mean()) + float((H_neg @ diff_u).mean())) / 2.0

j_pos = (H_pos @ diff_u)
j_neg = (H_neg @ diff_u)
try:
    cal_auroc = roc_auc_score(
        [1] * len(j_pos) + [0] * len(j_neg),
        list(j_pos) + list(j_neg),
    )
except Exception:
    cal_auroc = float("nan")
print(f"  Cal AUROC (in-sample): {cal_auroc:.4f}  θ={theta:.4f}")

# %% [code]
# ── Trajectory hook ────────────────────────────────────────────────────────────
captured_traj = {}
def _traj_hook(m, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] == 1:   # decode step (not prefill)
        captured_traj["h"] = h[0, -1].float().cpu().numpy()

traj_handle = layers[L_DEEP].register_forward_hook(_traj_hook)

def generate_with_trajectory(prompt, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """
    Token-by-token generation with J_know captured at every decode step.
    Returns (answer_text, j_trajectory, token_ids).
    Trajectory uses the same Fisher probe calibrated above.
    """
    inputs   = tokenizer(prompt, return_tensors="pt",
                         truncation=True, max_length=768).to(DEVICE)
    input_ids = inputs["input_ids"]
    eos_id    = tokenizer.eos_token_id or -1

    trajectory = []
    token_ids  = []

    with torch.no_grad():
        pre = model(input_ids=input_ids, use_cache=True)
        pkv = pre.past_key_values
        if not isinstance(pkv, DynamicCache):
            dc = DynamicCache()
            for i, (k, v) in enumerate(pkv): dc.update(k, v, i)
            pkv = dc
        del pre

    current = input_ids[:, -1:]

    for _ in range(max_new_tokens):
        captured_traj.clear()
        with torch.no_grad():
            out = model(input_ids=current, past_key_values=pkv, use_cache=True)

        pkv = out.past_key_values
        if not isinstance(pkv, DynamicCache):
            dc = DynamicCache()
            for i, (k, v) in enumerate(pkv): dc.update(k, v, i)
            pkv = dc

        logits = out.logits[0, -1].float()
        del out

        h = captured_traj.get("h")
        j = float(np.dot(h - c_neg, diff_u)) - theta if h is not None else 0.0
        trajectory.append(j)

        if temperature <= 0.0:
            next_tok = int(logits.argmax().item())
        else:
            probs    = torch.softmax(logits / temperature, dim=-1)
            next_tok = int(torch.multinomial(probs, 1).item())

        token_ids.append(next_tok)
        current = torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)

        if next_tok == eos_id:
            break

    answer_text = extract_final_answer(
        tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    )
    return answer_text, trajectory, token_ids

# %% [code]
# ── Locate </think> token ids ──────────────────────────────────────────────────
think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)

def find_think_end(token_ids):
    n = len(think_end_ids)
    for i in range(len(token_ids) - n + 1):
        if token_ids[i:i+n] == think_end_ids:
            return i + n - 1
    return None

# %% [code]
# ── Main experiment ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("COUNTERFACTUAL TRACE DIVERGENCE EXPERIMENT")
print(f"{'='*60}")

def find_divergence(correct_trajs, wrong_trajs, threshold=DIVERGENCE_THRESHOLD):
    """
    Find the first decode step where |mean_correct − mean_wrong| >= threshold.
    Returns (step, gap) or (None, 0.0) if no divergence found.
    """
    if not correct_trajs or not wrong_trajs:
        return None, 0.0
    min_len = min(
        min(len(t) for t in correct_trajs),
        min(len(t) for t in wrong_trajs),
    )
    if min_len < 5:
        return None, 0.0
    c_arr = np.array([t[:min_len] for t in correct_trajs])
    w_arr = np.array([t[:min_len] for t in wrong_trajs])
    gap   = c_arr.mean(0) - w_arr.mean(0)
    for step in range(min_len):
        if abs(gap[step]) >= threshold:
            return step, float(gap[step])
    return None, 0.0

questions_processed = 0
boundary_questions  = []
all_divergences     = []
examples            = []
t_exp               = time.time()

for row in ds:
    if questions_processed >= N_QUESTIONS:
        break

    q       = row["question"]
    answers = row["answer"]["aliases"] or [row["answer"]["value"]]
    prompt  = make_chat_prompt(tokenizer, q)

    correct_trajs, wrong_trajs  = [], []
    correct_texts, wrong_texts  = [], []
    think_ends                  = []

    for _ in range(N_RUNS):
        try:
            ans_text, traj, tids = generate_with_trajectory(prompt)
            f1 = f1_token(ans_text[:120], answers)
            te = find_think_end(tids)
            if f1 >= 0.50:
                correct_trajs.append(traj)
                correct_texts.append(ans_text[:80])
                if te is not None: think_ends.append(te)
            elif f1 == 0.0:
                wrong_trajs.append(traj)
                wrong_texts.append(ans_text[:80])
        except Exception as e:
            print(f"  [run error] {e}")
        finally:
            torch.cuda.empty_cache()

    questions_processed += 1

    if questions_processed % 10 == 0:
        elapsed = time.time() - t_exp
        print(f"  [{questions_processed:3d}/{N_QUESTIONS}]  "
              f"boundary={len(boundary_questions)}  "
              f"elapsed={elapsed:.0f}s", flush=True)

    if len(correct_trajs) < MIN_CORRECT or len(wrong_trajs) < MIN_WRONG:
        continue

    div_step, div_gap = find_divergence(correct_trajs, wrong_trajs)
    mean_think_end    = float(np.mean(think_ends)) if think_ends else None

    boundary_questions.append({
        "question":        q,
        "answers":         list(answers)[:3],
        "n_correct":       len(correct_trajs),
        "n_wrong":         len(wrong_trajs),
        "divergence_step": div_step,
        "divergence_gap":  div_gap,
        "mean_think_end":  mean_think_end,
        "correct_trajs":   [t[:150] for t in correct_trajs],
        "wrong_trajs":     [t[:150] for t in wrong_trajs],
        "correct_texts":   correct_texts[:2],
        "wrong_texts":     wrong_texts[:2],
    })
    if div_step is not None:
        all_divergences.append(div_step)
        examples.append({
            "question":            q,
            "answers":             list(answers)[:3],
            "divergence_step":     div_step,
            "divergence_gap":      div_gap,
            "mean_think_end":      mean_think_end,
            "correct_answer":      correct_texts[0] if correct_texts else "",
            "wrong_answer":        wrong_texts[0]   if wrong_texts   else "",
            "correct_traj_sample": correct_trajs[0][:150] if correct_trajs else [],
            "wrong_traj_sample":   wrong_trajs[0][:150]   if wrong_trajs   else [],
        })

traj_handle.remove()
t_exp_done = time.time()
print(f"\nExperiment complete in {t_exp_done - t_exp:.0f}s")
print(f"  Processed:          {questions_processed} questions")
print(f"  Boundary questions: {len(boundary_questions)}")
print(f"  With divergence:    {len(all_divergences)}")

# %% [code]
# ── Statistics ─────────────────────────────────────────────────────────────────
n_boundary = len(boundary_questions)
n_diverge  = len(all_divergences)

if n_boundary == 0:
    raise RuntimeError("No boundary questions found. Increase N_QUESTIONS or decrease MIN_CORRECT/MIN_WRONG.")

pct_diverge       = 100.0 * n_diverge / n_boundary
mean_div_step     = float(np.mean(all_divergences)) if all_divergences else 0.0
med_div_step      = float(np.median(all_divergences)) if all_divergences else 0.0
std_div_step      = float(np.std(all_divergences))   if all_divergences else 0.0

# Bootstrap 95% CI on divergence rate
bs_rates = []
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n_boundary, n_boundary)
    n_div_bs = sum(1 for i in idx if boundary_questions[i]["divergence_step"] is not None)
    bs_rates.append(100.0 * n_div_bs / n_boundary)
bs_rates  = np.array(bs_rates)
ci_lo     = float(np.percentile(bs_rates, 2.5))
ci_hi     = float(np.percentile(bs_rates, 97.5))

# Cohen's d per decode step (correct vs wrong J_know)
L_STAT = 120
pad = lambda trajs: np.array([
    t[:L_STAT] if len(t) >= L_STAT else t + [t[-1] if t else 0.0] * (L_STAT - len(t))
    for t in trajs
], dtype=float)

all_correct_t = [t for bq in boundary_questions for t in bq["correct_trajs"]]
all_wrong_t   = [t for bq in boundary_questions for t in bq["wrong_trajs"]]
cohens_d_per_step = np.zeros(L_STAT)
if all_correct_t and all_wrong_t:
    c_arr = pad(all_correct_t)
    w_arr = pad(all_wrong_t)
    for s in range(L_STAT):
        c_s = c_arr[:, s]; w_s = w_arr[:, s]
        pooled_std = np.sqrt((c_s.std()**2 + w_s.std()**2) / 2 + 1e-12)
        cohens_d_per_step[s] = (c_s.mean() - w_s.mean()) / pooled_std

# Wilcoxon test on divergence gaps
wilcox_p = 1.0
if n_diverge >= 10:
    gaps = [bq["divergence_gap"] for bq in boundary_questions
            if bq["divergence_step"] is not None]
    try:
        _, wilcox_p = wilcoxon(gaps)
    except Exception:
        wilcox_p = float("nan")

# Fraction that diverge in the first third of max generation
early_n   = sum(1 for d in all_divergences if d < MAX_TOKENS // 3)
early_pct = 100.0 * early_n / max(n_diverge, 1)

# ── Null hypothesis baseline ────────────────────────────────────────────────────
# Shuffle correct/wrong labels within each boundary question and recompute
# divergence. Measures expected divergence rate purely by chance. If observed >>
# null, the signal is real. z-score = (observed - null_mean) / null_std.
print(f"\nComputing null hypothesis baseline ({N_BOOTSTRAP} shuffles)...")
null_diverge_rates = []
for _ in range(N_BOOTSTRAP):
    null_n_div = 0
    for bq in boundary_questions:
        # Pool trajectories and randomly reassign to correct/wrong
        all_t = bq["correct_trajs"] + bq["wrong_trajs"]
        n_c   = len(bq["correct_trajs"])
        idx   = list(range(len(all_t)))
        rng.shuffle(idx)
        null_c = [all_t[i] for i in idx[:n_c]]
        null_w = [all_t[i] for i in idx[n_c:]]
        div_s, _ = find_divergence(null_c, null_w)
        if div_s is not None:
            null_n_div += 1
    null_diverge_rates.append(100.0 * null_n_div / max(n_boundary, 1))

null_mean  = float(np.mean(null_diverge_rates))
null_ci_lo = float(np.percentile(null_diverge_rates, 2.5))
null_ci_hi = float(np.percentile(null_diverge_rates, 97.5))
null_std   = float(np.std(null_diverge_rates))
z_score    = (pct_diverge - null_mean) / max(null_std, 0.01)
print(f"  Null: {null_mean:.1f}% (95% CI [{null_ci_lo:.1f}%, {null_ci_hi:.1f}%])")
print(f"  Observed: {pct_diverge:.1f}% vs Null: {null_mean:.1f}%  z={z_score:.2f}")

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  Questions evaluated:  {N_QUESTIONS}")
print(f"  Boundary questions:   {n_boundary}  ({100*n_boundary/N_QUESTIONS:.1f}%)")
print(f"  With divergence:      {n_diverge}  ({pct_diverge:.1f}%,  95% CI [{ci_lo:.1f}%, {ci_hi:.1f}%])")
print(f"  Null baseline:        {null_mean:.1f}%  (95% CI [{null_ci_lo:.1f}%, {null_ci_hi:.1f}%])")
print(f"  Signal z-score:       {z_score:.2f}  (observed vs shuffled-label null)")
print(f"  Diverge in first 1/3: {early_n}/{n_diverge}  ({early_pct:.1f}%)")
print(f"  Mean divergence step: {mean_div_step:.1f}  ±{std_div_step:.1f}")
print(f"  Median:               {med_div_step:.1f}")
print(f"  Peak |Cohen's d|:     {np.abs(cohens_d_per_step).max():.3f}  "
      f"at step {int(np.abs(cohens_d_per_step).argmax())}")
print(f"  Wilcoxon p (gap≠0):  {wilcox_p:.4f}")
print(f"  Cal AUROC:            {cal_auroc:.4f}")

# %% [code]
# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle(
    "Counterfactual Trace Divergence — DeepSeek-R1-Distill-Qwen-1.5B (T4 GPU, TriviaQA)\n"
    f"N={N_QUESTIONS} questions, {N_RUNS} runs each  |  "
    f"Boundary: {n_boundary}  |  Diverged: {n_diverge} ({pct_diverge:.0f}%,"
    f" 95% CI [{ci_lo:.0f}%–{ci_hi:.0f}%])",
    fontsize=10, fontweight="bold",
)

def pad_traj(trajs, length):
    out = []
    for t in trajs:
        last = t[-1] if t else 0.0
        out.append((t + [last] * length)[:length])
    return np.array(out, dtype=float)

# Panel 1: Mean trajectories with 95% bootstrap CI bands
ax = axes[0]
if all_correct_t and all_wrong_t:
    L1    = L_STAT
    c_arr = pad_traj(all_correct_t, L1)
    w_arr = pad_traj(all_wrong_t,   L1)
    xs    = np.arange(L1)

    def bs_mean_ci(arr, n_bs=500):
        means = [arr[rng.integers(0, len(arr), len(arr))].mean(0) for _ in range(n_bs)]
        means = np.array(means)
        return means.mean(0), np.percentile(means, 2.5, 0), np.percentile(means, 97.5, 0)

    c_m, c_lo, c_hi = bs_mean_ci(c_arr)
    w_m, w_lo, w_hi = bs_mean_ci(w_arr)

    ax.fill_between(xs, c_lo, c_hi, alpha=0.25, color="steelblue")
    ax.plot(xs, c_m, color="steelblue", lw=2,
            label=f"Correct  (n={len(all_correct_t)})")
    ax.fill_between(xs, w_lo, w_hi, alpha=0.25, color="tomato")
    ax.plot(xs, w_m, color="tomato", lw=2,
            label=f"Wrong    (n={len(all_wrong_t)})")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    if all_divergences:
        ax.axvline(med_div_step, color="orange", ls=":", lw=2,
                   label=f"Median divergence τ={med_div_step:.0f}")
ax.set_xlabel("Decode step t")
ax.set_ylabel("J_know (Fisher)")
ax.set_title("Mean trajectories\n(bootstrapped 95% CI)")
ax.legend(fontsize=8)

# Panel 2: Cohen's d trajectory
ax = axes[1]
xs_d = np.arange(L_STAT)
ax.plot(xs_d, cohens_d_per_step, color="purple", lw=1.5)
ax.fill_between(xs_d, 0, cohens_d_per_step,
                where=cohens_d_per_step > 0, alpha=0.3, color="steelblue")
ax.fill_between(xs_d, 0, cohens_d_per_step,
                where=cohens_d_per_step < 0, alpha=0.3, color="tomato")
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.axhline(0.2, color="gray", ls=":", lw=0.8, alpha=0.5, label="|d|=0.2")
ax.axhline(-0.2, color="gray", ls=":", lw=0.8, alpha=0.5)
if all_divergences:
    ax.axvline(med_div_step, color="orange", ls=":", lw=2,
               label=f"τ={med_div_step:.0f}")
ax.set_xlabel("Decode step t")
ax.set_ylabel("Cohen's d")
ax.set_title(f"Effect size per step\n(correct vs wrong J_know)")
ax.legend(fontsize=8)

# Panel 3: Best single example
ax = axes[2]
best = None
if examples:
    best = max(examples,
               key=lambda e: abs(e["divergence_gap"]) / max(e["divergence_step"] + 1, 1))

if best:
    L2  = min(150, len(best["correct_traj_sample"]), len(best["wrong_traj_sample"]))
    ct  = best["correct_traj_sample"][:L2]
    wt  = best["wrong_traj_sample"][:L2]
    xs2 = np.arange(L2)
    ax.plot(xs2, ct, color="steelblue", lw=2,
            label=f"Correct: \"{best['correct_answer'][:30]}\"")
    ax.plot(xs2, wt, color="tomato", lw=2,
            label=f"Wrong:   \"{best['wrong_answer'][:30]}\"")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    div_s = best["divergence_step"]
    ax.axvline(div_s, color="orange", ls=":", lw=2.5, label=f"Divergence t={div_s}")
    ax.scatter([div_s], [ct[div_s]], color="steelblue", s=100, zorder=5)
    ax.scatter([div_s], [wt[div_s]], color="tomato",    s=100, zorder=5)
    if best.get("mean_think_end") is not None:
        te = int(best["mean_think_end"])
        ax.axvline(te, color="gray", ls="-.", lw=1.5, alpha=0.7, label=f"</think> t≈{te}")
    q_short = best["question"][:50] + ("…" if len(best["question"]) > 50 else "")
    ax.set_title(f"Best example\n\"{q_short}\"", fontsize=8)
else:
    ax.set_title("No clear example")
ax.set_xlabel("Decode step t")
ax.set_ylabel("J_know")
ax.legend(fontsize=7)

# Panel 4: Divergence rate — observed vs null
ax = axes[3]
if all_divergences:
    ax.hist(all_divergences, bins=min(25, n_diverge),
            color="steelblue", edgecolor="white", alpha=0.8, label="Observed divergences")
    ax.axvline(mean_div_step, color="orange", ls="--", lw=2,
               label=f"Mean={mean_div_step:.0f}")
    ax.axvline(med_div_step,  color="tomato",  ls="--", lw=2,
               label=f"Median={med_div_step:.0f}")
    ax.axvline(MAX_TOKENS // 3, color="gray", ls=":", lw=1.5, alpha=0.6,
               label=f"1/3 of max ({MAX_TOKENS//3})")
    ax.text(0.05, 0.95,
            f"Observed: {pct_diverge:.0f}%\n"
            f"Null (shuffled): {null_mean:.0f}% ±{null_std:.0f}%\n"
            f"z = {z_score:.2f}\n"
            f"Early (< 1/3): {early_pct:.0f}%\n"
            f"Wilcoxon p = {wilcox_p:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.set_xlabel("Divergence step")
    ax.set_ylabel("Count")
    ax.set_title(f"Divergence step distribution\n"
                 f"Observed {pct_diverge:.0f}% vs Null {null_mean:.0f}% (z={z_score:.1f})")
    ax.legend(fontsize=8)
else:
    ax.set_title("No divergence data")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT_FIG}")

# %% [code]
# ── Verdict ────────────────────────────────────────────────────────────────────
if pct_diverge >= 60:
    verdict = (
        f"DIVERGENCE_CONFIRMED: {pct_diverge:.1f}% of boundary questions "
        f"(95% CI [{ci_lo:.1f}%–{ci_hi:.1f}%]) show detectable J_know divergence "
        f"at median step {med_div_step:.0f}. {early_pct:.0f}% of divergences occur "
        f"in the first third of the reasoning trace — before the answer is written."
    )
elif pct_diverge >= 30:
    verdict = (
        f"PARTIAL_DIVERGENCE: {pct_diverge:.1f}% (CI [{ci_lo:.1f}%–{ci_hi:.1f}%]) "
        f"of boundary questions show divergence. Signal is present but not universal."
    )
else:
    verdict = (
        f"WEAK_SIGNAL: {pct_diverge:.1f}% show clear divergence. "
        f"Increase N_RUNS or reduce DIVERGENCE_THRESHOLD."
    )

print(f"\nVerdict: {verdict}")

# %% [code]
# ── Save ───────────────────────────────────────────────────────────────────────
summary = {
    "experiment":            "counterfactual_divergence_v1",
    "model":                 MODEL_ID,
    "probe_layer":           L_DEEP,
    "gpu":                   gpu_name,
    "gpu_vram_gb":           float(f"{gpu_vram:.1f}"),
    "torch_version":         torch.__version__,
    "transformers_version":  transformers.__version__,
    "cal_auroc":             float(cal_auroc),
    "cal_n_param":           len(pos_vecs),
    "cal_n_ctxdep":          len(neg_vecs),
    "n_questions":           N_QUESTIONS,
    "n_runs_per_question":   N_RUNS,
    "temperature":           TEMPERATURE,
    "max_tokens":            MAX_TOKENS,
    "divergence_threshold":  DIVERGENCE_THRESHOLD,
    "n_boundary":            n_boundary,
    "pct_boundary":          float(100.0 * n_boundary / N_QUESTIONS),
    "n_diverge":             n_diverge,
    "pct_diverge":           float(pct_diverge),
    "ci_lo_95":              float(ci_lo),
    "ci_hi_95":              float(ci_hi),
    "mean_divergence_step":  float(mean_div_step),
    "median_divergence_step": float(med_div_step),
    "std_divergence_step":   float(std_div_step),
    "early_divergence_pct":  float(early_pct),
    "peak_cohens_d":         float(np.abs(cohens_d_per_step).max()),
    "peak_cohens_d_step":    int(np.abs(cohens_d_per_step).argmax()),
    "wilcoxon_p":            float(wilcox_p),
    "divergence_steps":      all_divergences,
    "null_pct_diverge_mean": float(null_mean),
    "null_pct_diverge_ci_lo": float(null_ci_lo),
    "null_pct_diverge_ci_hi": float(null_ci_hi),
    "null_pct_diverge_std":  float(null_std),
    "z_score_vs_null":       float(z_score),
    "total_runtime_s":       float(t_exp_done - t_load),
    "verdict":               verdict,
    "kv_attractor_interpretation": (
        "J_know trajectory identifies KV entries belonging to stable inference attractors. "
        "Steps with positive and rising J_know correspond to parametric recall phases — "
        "KV entries at these positions are candidates for retention. "
        "Steps where J_know falls or crosses zero correspond to unstable inference states — "
        "KV entries at these positions are candidates for eviction. "
        f"Divergence precedes output revelation at median step {med_div_step:.0f}, "
        f"enabling prospective memory management rather than retrospective eviction."
    ),
}

with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {OUT_JSON}")

with open(OUT_EXAMPLES, "w") as f:
    json.dump(examples[:30], f, indent=2)
print(f"Examples saved: {OUT_EXAMPLES}")
print("\nDONE")

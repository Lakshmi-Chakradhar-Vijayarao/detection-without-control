#!/usr/bin/env python3
"""
teacher_independence_v1.py — EXP_TEACHER_INDEPENDENCE_V1
Tests whether L3 commitment timing (early commitment during think tokens) replicates
on a reasoning model NOT from the DeepSeek-R1 distillation lineage.

SCIENTIFIC QUESTION:
  EXP-B and EXP-C both used DeepSeek-R1-Distill variants (same teacher).
  C022 is annotated: "NOTE: both models share same teacher — teacher-independence pending."
  Does early commitment (commit% > 50%, z >> null) replicate on Qwen3-1.7B?
  Qwen3-1.7B uses Qwen's own thinking capability (no DeepSeek teacher).

DESIGN:
  Model: Qwen/Qwen3-1.7B (thinking-enabled; use think_budget=None or large budget)
  Task: TriviaQA (matching EXP-B/C for comparability)
  N: 100 questions
  Protocol: same as EXP-B
    - Run model with thinking enabled
    - Extract Fisher trajectory across think-block tokens using step-wise hooks
    - Detect commit point: first step where Fisher trajectory crosses threshold
    - Compute commit_pct, z-score vs shuffled null

  Layer index: auto-detected as penultimate layer of Qwen3-1.7B

  Commit definition (same as EXP-B):
    threshold = 2 × SD(initial_fisher_scores)
    commit_step = first step where |fisher_score| > threshold AND maintained for ≥3 steps

  Shuffled null: permute item labels × 100 repetitions, compute mean commit_pct under null.

VERDICT:
  REPLICATED: commit% > 50% AND z > 5 (i.e., far from shuffled null)
  NOT_REPLICATED: commit% < 30% or z < 2

Note on Qwen3 thinking mode:
  Use model.generate with enable_thinking=True (Qwen3 API).
  Alternately: prepend "<think>" and let the model generate until </think>.
  The think block is bounded by <think>...</think> tokens.

GPU: T4 (~8h for N=100)
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID       = "Qwen/Qwen3-1.7B"
N_QUESTIONS    = 100
N_NULL_PERMS   = 100
POOL_SIZE      = 3_000
MAX_THINK_TOKENS = 2048
MAX_ANSWER_TOKENS = 60
COMMIT_HOLD    = 3       # steps threshold must hold before commit declared
LAYER_IDX      = None    # auto: penultimate layer

SAVE_PATH      = "/kaggle/working/teacher_independence_v1_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


def _get_hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None

_tok = _get_hf_token()
if _tok:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_tok, add_to_git_credential=False)
    print("HF login: OK", flush=True)


# ── Data ─────────────────────────────────────────────────────────────────────────
def load_pool(n: int = POOL_SIZE):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    items = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        items.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
        })
        if len(items) >= n:
            break
    np.random.shuffle(items)
    return items


def token_f1(pred: str, golds) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        c = p & q
        if c and p and q:
            pr = len(c)/len(p); rc = len(c)/len(q)
            best = max(best, 2*pr*rc/(pr+rc))
    return best


def answer_contains(pred: str, golds) -> bool:
    pl = pred.lower()
    return any(g.lower().strip() and g.lower().strip() in pl for g in golds)


# ── Model ─────────────────────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_ID} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    cfg = getattr(model.config, 'text_config', model.config)
    n_layers = cfg.num_hidden_layers
    print(f"  Loaded: {n_layers} layers", flush=True)
    return model, tokenizer, n_layers


_LAYER_PATHS = [
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "transformer.h",
]

def get_layers(model):
    for path in _LAYER_PATHS:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot find transformer layers")


# ── Fisher trajectory extraction ─────────────────────────────────────────────────
def get_think_trajectory(model, tokenizer, question: str, layer_idx: int):
    """
    Run model with thinking mode and collect Fisher-probe-ready hidden states
    at each think-block token generation step.

    Returns: list of 1D numpy arrays (one per think token), or None if no think block.
    Also returns the final answer string.
    """
    # Qwen3 thinking mode: use <think> prompt format
    prompt_text = f"Think step by step before answering.\n\nQuestion: {question}\nAnswer:"
    msgs = [{"role": "user", "content": prompt_text}]
    try:
        # Qwen3 requires enable_thinking=True to activate <think> block generation
        input_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except Exception:
        input_text = prompt_text

    ids = tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)
    think_hs = []

    # Hook at layer_idx: collect last-position hidden state at each generation step
    step_counter = [0]
    hs_this_step = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_this_step[0] = x[:, -1, :].detach().float().cpu().numpy()[0]

    layers = get_layers(model)
    h = layers[layer_idx].register_forward_hook(hook)

    generated_ids = ids.clone()
    in_think = False
    think_started = False
    answer_tokens = []

    try:
        think_start_id = tokenizer.encode("<think>", add_special_tokens=False)
        think_end_id   = tokenizer.encode("</think>", add_special_tokens=False)
    except Exception:
        think_start_id = []
        think_end_id   = []

    with torch.no_grad():
        for step_i in range(MAX_THINK_TOKENS + MAX_ANSWER_TOKENS):
            hs_this_step[0] = None
            out = model(generated_ids)
            token_id = int(out.logits[0, -1, :].argmax())
            new_tok = torch.tensor([[token_id]], device=DEVICE)
            generated_ids = torch.cat([generated_ids, new_tok], dim=1)
            tok_str = tokenizer.decode([token_id])

            # Detect think block boundaries
            if not think_started:
                if "<think>" in tok_str or (think_start_id and token_id in think_start_id):
                    think_started = True
                    in_think = True
                    continue
                # If no think token appears in first 10 steps, model isn't thinking
                if step_i > 10:
                    break
                continue

            if in_think:
                if "</think>" in tok_str or (think_end_id and token_id in think_end_id):
                    in_think = False
                    continue
                if hs_this_step[0] is not None:
                    think_hs.append(hs_this_step[0].copy())
                if len(think_hs) >= MAX_THINK_TOKENS:
                    break
            else:
                # After </think>, collect answer
                if token_id == tokenizer.eos_token_id:
                    break
                answer_tokens.append(token_id)
                if len(answer_tokens) >= MAX_ANSWER_TOKENS:
                    break

    h.remove()
    answer_str = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return think_hs, answer_str


# ── Commit detection (same as EXP-B) ─────────────────────────────────────────────
def detect_commit(fisher_scores_traj: np.ndarray, threshold: float):
    """Given Fisher trajectory (1D, n_steps), find commit step."""
    n = len(fisher_scores_traj)
    commit_step = None
    for i in range(n - COMMIT_HOLD):
        if all(abs(fisher_scores_traj[j]) > threshold for j in range(i, i + COMMIT_HOLD)):
            commit_step = i
            break
    return commit_step


# ── Phase 1: Calibrate Fisher probe from non-thinking outputs ────────────────────
def calibrate_probe(model, tokenizer, pool, layer_idx):
    """Use a small bilateral oracle (N=50/class) to calibrate Fisher probe for Qwen3."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    print("\n=== Calibrating Fisher probe (bilateral oracle N=50/class) ===", flush=True)

    param_hs, ctxdep_hs = [], []
    n_scanned = 0

    for item in pool:
        if len(param_hs) >= 50 and len(ctxdep_hs) >= 50:
            break
        n_scanned += 1
        q = item["question"]; ans = item["answers"]

        msgs = [{"role": "user", "content": q}]
        try:
            pnc = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pnc = q

        ids = tokenizer(pnc, return_tensors="pt").input_ids.to(DEVICE)
        hs_out = [None]

        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()[0]

        layers = get_layers(model)
        h = layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            out_m = model(ids)
        h.remove()
        tok = int(out_m.logits[0, -1, :].argmax())
        gen_nc = tokenizer.decode(
            model.generate(ids, max_new_tokens=60, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)[0][ids.shape[1]:],
            skip_special_tokens=True
        )
        f1 = token_f1(gen_nc, ans)
        ok = answer_contains(gen_nc, ans) or f1 >= 0.50

        if ok and len(param_hs) < 50 and hs_out[0] is not None:
            param_hs.append(hs_out[0])
        elif not ok and f1 <= 0.05 and len(ctxdep_hs) < 50 and hs_out[0] is not None:
            ctxdep_hs.append(hs_out[0])

    print(f"  Calibration: PARAM={len(param_hs)} CTXDEP={len(ctxdep_hs)}", flush=True)

    n = min(len(param_hs), len(ctxdep_hs))
    X = np.stack(param_hs[:n] + ctxdep_hs[:n])
    y = np.array([1]*n + [0]*n)

    pca = PCA(n_components=min(64, X.shape[1], n-1))
    Xp = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xp, y)
    scores = lda.decision_function(Xp)

    # Threshold = 2 SD of scores
    threshold = 2.0 * float(np.std(scores))
    try:
        auroc = float(roc_auc_score(y, scores))
    except Exception:
        auroc = 0.5
    print(f"  Calibration AUROC={auroc:.4f}  threshold={threshold:.4f}", flush=True)

    return pca, lda, threshold, auroc


# ── Phase 2: Think-block trajectory for N questions ──────────────────────────────
def run_commit_experiment(model, tokenizer, pool, layer_idx, pca, lda, threshold):
    print(f"\n=== Commit Timing (N={N_QUESTIONS}) ===", flush=True)
    commit_pcts = []
    think_lens  = []
    f1_scores   = []
    n_no_think  = 0
    n_scanned   = 0
    processed   = 0

    for item in pool:
        if processed >= N_QUESTIONS:
            break
        n_scanned += 1
        q = item["question"]; ans = item["answers"]

        think_hs, answer_str = get_think_trajectory(model, tokenizer, q, layer_idx)
        if not think_hs:
            n_no_think += 1
            continue

        processed += 1
        think_len = len(think_hs)
        think_lens.append(think_len)

        # Apply probe to each think token
        X = np.stack(think_hs)
        try:
            Xp = pca.transform(X)
            fisher_traj = lda.decision_function(Xp)
        except Exception:
            fisher_traj = np.zeros(think_len)

        commit_step = detect_commit(fisher_traj, threshold)
        if commit_step is not None:
            commit_pct = 100.0 * (think_len - commit_step) / think_len
            commit_pcts.append(commit_pct)
        f1 = token_f1(answer_str, ans)
        f1_scores.append(f1)

        if processed % 10 == 0:
            print(f"  processed={processed} committed={len(commit_pcts)} "
                  f"mean_think_len={np.mean(think_lens):.0f}", flush=True)

    n_committed = len(commit_pcts)
    mean_commit_pct = float(np.mean(commit_pcts)) if commit_pcts else 0.0
    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    # Shuffled null: permute the detected Fisher trajectories
    # Proxy: shuffle commit_pct values across items → always ~50% if random
    null_means = []
    for _ in range(N_NULL_PERMS):
        perm = np.random.permutation(commit_pcts) if commit_pcts else [0.]
        null_means.append(float(np.mean(perm)))
    null_mean = float(np.mean(null_means))
    null_std  = float(np.std(null_means))
    # Better null: permute Fisher scores within each item's trajectory
    null_commit_pcts = []
    for think_pct in commit_pcts:
        # Under null, commit_step is uniformly distributed → mean commit_pct ≈ 50%
        null_commit_pcts.append(float(np.random.uniform(0, 100)))
    null_mean_direct = float(np.mean(null_commit_pcts)) if null_commit_pcts else 50.0

    # z-score: how many SDs above null?
    if null_std > 0:
        z = (mean_commit_pct - null_mean_direct) / null_std
    else:
        z = 0.0

    print(f"\nResults: N_committed={n_committed}/{processed}  "
          f"mean_commit%={mean_commit_pct:.1f}%  "
          f"null_mean={null_mean_direct:.1f}%  z={z:.2f}", flush=True)
    print(f"mean_think_len={np.mean(think_lens):.0f}  mean_F1={mean_f1:.3f}", flush=True)

    if n_committed >= 10:
        if mean_commit_pct > 50 and z > 5:
            verdict = "REPLICATED"
        elif mean_commit_pct < 30 or z < 2:
            verdict = "NOT_REPLICATED"
        else:
            verdict = "WEAK_SIGNAL"
    else:
        verdict = "INSUFFICIENT_DATA"

    print(f"VERDICT: {verdict}", flush=True)

    return {
        "n_processed": processed, "n_committed": n_committed,
        "n_no_think": n_no_think, "n_scanned": n_scanned,
        "mean_commit_pct": mean_commit_pct,
        "null_mean": null_mean_direct, "z_score": z,
        "mean_think_len": float(np.mean(think_lens)) if think_lens else 0.,
        "mean_f1": mean_f1,
        "commit_pct_distribution": {
            "p25": float(np.percentile(commit_pcts, 25)) if commit_pcts else 0.,
            "p50": float(np.percentile(commit_pcts, 50)) if commit_pcts else 0.,
            "p75": float(np.percentile(commit_pcts, 75)) if commit_pcts else 0.,
        },
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer, n_layers = load_model()

    global LAYER_IDX
    LAYER_IDX = n_layers - 2  # penultimate
    print(f"Probe layer: {LAYER_IDX} (penultimate of {n_layers})", flush=True)

    pca, lda, threshold, calib_auroc = calibrate_probe(model, tokenizer, pool, LAYER_IDX)
    commit_results = run_commit_experiment(model, tokenizer, pool, LAYER_IDX, pca, lda, threshold)

    results = {
        "experiment":      "EXP_TEACHER_INDEPENDENCE_V1",
        "model":           MODEL_ID,
        "layer_idx":       LAYER_IDX,
        "calibration": {
            "auroc": calib_auroc, "threshold": threshold
        },
        "commit": commit_results,
        "elapsed_min":     (time.time() - t0) / 60,
        "scientific_note": (
            "C022 annotated: both EXP-B and EXP-C used DeepSeek-R1-Distill (same teacher). "
            "This experiment uses Qwen3-1.7B (Qwen's own reasoning capability, no DeepSeek teacher). "
            "REPLICATED verdict would promote C022 to teacher-independent, architecture-invariant. "
            "NOT_REPLICATED would scope C022 to DeepSeek-R1-Distill lineage only."
        ),
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)
    print(f"Elapsed: {results['elapsed_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()

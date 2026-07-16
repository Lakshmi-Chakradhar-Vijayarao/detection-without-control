"""
EXP-I V2: Early Exit Causal Truncation — N=500, Formal Statistics
===================================================================
Scales V1 from N=200 to N=500 to achieve two-tailed p<0.05.
At V1 SE=0.0034 (N=200), SE at N=500 ≈ 0.0021 → t=2.81 → p≈0.005 (if effect holds).

Key additions over V1:
  - N_QUESTIONS = 500
  - Formal two-tailed t-test + Wilcoxon signed-rank test
  - Bootstrapped 95% CI on mean delta
  - Intermediate saves every 50 items

Design: For each question, generate TWICE:
  1. Full reasoning: complete think block, extract answer
  2. Truncated: Fisher probe detects commit point, inject </think> there

Primary metric: mean Δf1 = f1_full - f1_truncated across committed items.
Null hypothesis (H0): mean Δf1 = 0.
"""

import gc, json, os, random, time
import numpy as np
import torch
from datasets import load_dataset
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID         = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
THINK_END_STR    = "</think>"

POOL_SIZE        = 10000
N_QUESTIONS      = 500
LAYER_IDX        = 26
PCA_DIM          = 64
SEED             = 42
MAX_THINK_TOKENS = 800
MAX_ANS_TOKENS   = 80
COMMIT_WINDOW    = 5
COMMIT_PERSIST   = 3
CALIBRATION_N    = 60   # items for Fisher probe calibration
N_BOOTSTRAP      = 2000  # bootstrap resamples for CI

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required — run on T4")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Config: N={N_QUESTIONS}  POOL={POOL_SIZE}  LAYER={LAYER_IDX}  CAL_N={CALIBRATION_N}", flush=True)

rng = random.Random(SEED)
np.random.seed(SEED)


# ── Utilities ─────────────────────────────────────────────────────────────────
def _get_hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k, "")
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def token_f1(pred: str, golds: list) -> float:
    pred_t = pred.lower().split()
    best = 0.0
    for g in golds:
        g_t = g.lower().split()
        common = set(pred_t) & set(g_t)
        if not common:
            continue
        p = len(common) / len(pred_t) if pred_t else 0
        r = len(common) / len(g_t) if g_t else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        best = max(best, f1)
    return best


def answer_contains(pred: str, golds: list) -> bool:
    return any(g.lower() in pred.lower() for g in golds)


def fmt_prompt(q: str) -> str:
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def load_pool(n: int):
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
    pool = []
    for ex in ds:
        if len(pool) >= n:
            break
        q   = ex["question"]
        ans = ex["answer"]["aliases"][:5]
        if q and ans:
            pool.append({"q": q, "answers": ans})
    rng.shuffle(pool)
    return pool


def bootstrap_ci(deltas: list, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple:
    arr = np.array(deltas)
    boot_means = np.array([
        np.mean(arr[np.random.randint(0, len(arr), len(arr))])
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Hook helpers ──────────────────────────────────────────────────────────────
def extract_step1_hidden(mdl, tok, prompt: str) -> np.ndarray | None:
    hs_store = {}

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:
            hs_store["hs"] = hs[0, 0, :].float().cpu().numpy()

    h = mdl.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
    try:
        inp_t = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            _ = mdl.generate(**inp_t, max_new_tokens=1, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        h.remove()
    return hs_store.get("hs")


# ── Calibration: build Fisher probe ──────────────────────────────────────────
def calibrate_probe(mdl, tok, pool):
    """Build committed/uncommitted probe from early vs late think-step hidden states."""
    committed_hs = []
    uncommitted_hs = []
    t0 = time.time()

    for item in pool[:CALIBRATION_N * 15]:
        if len(committed_hs) >= CALIBRATION_N and len(uncommitted_hs) >= CALIBRATION_N:
            break
        q = item["q"]
        prompt = fmt_prompt(q)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        think_end_id = tok.encode(THINK_END_STR, add_special_tokens=False)[-1]
        hs_list = []
        hs_store = {}

        def hook_fn(module, inp_h, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] == 1:
                hs_store["cur"] = hs[0, 0, :].float().cpu().numpy()

        h = mdl.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                pre = mdl(inp["input_ids"], use_cache=True)
                pkv = pre.past_key_values
            cur_ids = inp["input_ids"]
            for step in range(120):
                with torch.no_grad():
                    out = mdl(cur_ids[:, -1:], past_key_values=pkv, use_cache=True)
                logits = out.logits[0, -1, :]
                next_id = int(logits.argmax().item())
                pkv = out.past_key_values
                cur_ids = torch.tensor([[next_id]], device=DEVICE)
                cur_hs = hs_store.get("cur")
                if cur_hs is not None:
                    hs_list.append((step, cur_hs))
                if next_id == think_end_id or step >= 119:
                    break
        finally:
            h.remove()

        if len(hs_list) < 10:
            continue
        n_each = min(3, len(hs_list) // 3)
        for _, hs in hs_list[:n_each]:
            if len(uncommitted_hs) < CALIBRATION_N:
                uncommitted_hs.append(hs)
        for _, hs in hs_list[-n_each:]:
            if len(committed_hs) < CALIBRATION_N:
                committed_hs.append(hs)

    if len(committed_hs) < 10 or len(uncommitted_hs) < 10:
        return None, None, None

    n_min = min(len(committed_hs), len(uncommitted_hs))
    X = np.array(committed_hs[:n_min] + uncommitted_hs[:n_min])
    y = np.array([1] * n_min + [0] * n_min)
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_r = pca.fit_transform(X.astype(np.float32))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_r, y)

    com_scores = lda.decision_function(pca.transform(
        np.array(committed_hs[:n_min]).astype(np.float32)))
    thresh = float(np.percentile(com_scores, 20))
    print(f"Calibration: n={n_min}/class  thresh={thresh:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)
    return pca, lda, thresh


# ── Full generation ───────────────────────────────────────────────────────────
def generate_full(mdl, tok, question: str) -> tuple:
    """Returns (answer, think_len, per-step hidden states)."""
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    think_end_id = tok.encode(THINK_END_STR, add_special_tokens=False)[-1]
    hs_by_step = []
    hs_store = {}

    def hook_fn(module, inp_h, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:
            hs_store["cur"] = hs[0, 0, :].float().cpu().numpy()

    h = mdl.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            pre = mdl(inp["input_ids"], use_cache=True)
            pkv = pre.past_key_values
        cur_ids = inp["input_ids"]
        think_len = 0
        ans_tokens = []
        in_answer = False

        for step in range(MAX_THINK_TOKENS + MAX_ANS_TOKENS):
            with torch.no_grad():
                out = mdl(cur_ids[:, -1:], past_key_values=pkv, use_cache=True)
            next_id = int(out.logits[0, -1, :].argmax().item())
            pkv = out.past_key_values
            cur_ids = torch.tensor([[next_id]], device=DEVICE)

            if not in_answer:
                cur_hs = hs_store.get("cur")
                if cur_hs is not None:
                    hs_by_step.append(cur_hs.copy())
                if next_id == think_end_id:
                    think_len = step + 1
                    in_answer = True
            else:
                ans_tokens.append(next_id)
                if next_id == tok.eos_token_id or len(ans_tokens) >= MAX_ANS_TOKENS:
                    break
    finally:
        h.remove()

    answer = tok.decode(ans_tokens, skip_special_tokens=True).strip()
    return answer, think_len, hs_by_step


# ── Truncated generation ──────────────────────────────────────────────────────
def generate_truncated(mdl, tok, question: str, commit_step: int) -> str:
    """Truncate at commit_step, inject </think>, generate answer."""
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    think_end_id = tok.encode(THINK_END_STR, add_special_tokens=False)[-1]

    with torch.no_grad():
        pre = mdl(inp["input_ids"], use_cache=True)
        pkv = pre.past_key_values
    cur_ids = inp["input_ids"]

    for step in range(commit_step):
        with torch.no_grad():
            out = mdl(cur_ids[:, -1:], past_key_values=pkv, use_cache=True)
        next_id = int(out.logits[0, -1, :].argmax().item())
        pkv = out.past_key_values
        cur_ids = torch.tensor([[next_id]], device=DEVICE)

    think_ids = torch.tensor([[think_end_id]], device=DEVICE)
    with torch.no_grad():
        out = mdl(think_ids, past_key_values=pkv, use_cache=True)
    pkv = out.past_key_values
    first_ans_id = int(out.logits[0, -1, :].argmax().item())

    ans_tokens = [first_ans_id]
    cur_ids = torch.tensor([[first_ans_id]], device=DEVICE)
    if first_ans_id != tok.eos_token_id:
        for _ in range(MAX_ANS_TOKENS - 1):
            with torch.no_grad():
                out = mdl(cur_ids, past_key_values=pkv, use_cache=True)
            next_id = int(out.logits[0, -1, :].argmax().item())
            pkv = out.past_key_values
            cur_ids = torch.tensor([[next_id]], device=DEVICE)
            ans_tokens.append(next_id)
            if next_id == tok.eos_token_id:
                break

    return tok.decode(ans_tokens, skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("EXP-I V2: Early Exit Causal Truncation (N=500)", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)

    hf_token = _get_hf_token()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True, token=hf_token
    ).to(DEVICE).eval()

    pool = load_pool(POOL_SIZE)

    print("\nPhase 1: Calibrating Fisher probe...", flush=True)
    pca, lda, thresh = calibrate_probe(mdl, tok, pool)
    if pca is None:
        raise RuntimeError("Calibration failed")

    print(f"\nPhase 2: Dual-run experiment (N={N_QUESTIONS})...", flush=True)
    records = []
    t0 = time.time()

    for i, item in enumerate(pool[CALIBRATION_N * 15:]):
        if len(records) >= N_QUESTIONS:
            break
        q, ans = item["q"], item["answers"]

        try:
            answer_full, think_len, hs_traj = generate_full(mdl, tok, q)
        except Exception as e:
            print(f"  [SKIP full] {e}", flush=True)
            continue

        if think_len == 0 or len(hs_traj) < COMMIT_WINDOW + COMMIT_PERSIST + 1:
            continue

        hs_arr = np.array(hs_traj).astype(np.float32)
        X_pca = pca.transform(hs_arr)
        scores = lda.decision_function(X_pca).tolist()

        commit_step = None
        for s in range(COMMIT_WINDOW, len(scores) - COMMIT_PERSIST):
            if scores[s] > thresh:
                if all(scores[s + j] > thresh for j in range(COMMIT_PERSIST)):
                    commit_step = s
                    break

        if commit_step is None:
            f1_full = token_f1(answer_full, ans)
            records.append({
                "q": q, "commit_step": None, "think_len": think_len,
                "f1_full": f1_full, "f1_trunc": None, "delta_f1": None,
                "committed": False
            })
            continue

        try:
            answer_trunc = generate_truncated(mdl, tok, q, commit_step)
        except Exception as e:
            print(f"  [SKIP trunc] {e}", flush=True)
            continue

        f1_full  = token_f1(answer_full, ans)
        f1_trunc = token_f1(answer_trunc, ans)
        commit_pct = 100.0 * (think_len - commit_step) / think_len

        records.append({
            "q": q,
            "commit_step": commit_step,
            "think_len": think_len,
            "commit_pct": commit_pct,
            "f1_full": f1_full,
            "f1_trunc": f1_trunc,
            "delta_f1": f1_full - f1_trunc,
            "answer_full": answer_full,
            "answer_trunc": answer_trunc,
            "committed": True
        })

        if len(records) % 50 == 0:
            committed = [r for r in records if r["committed"]]
            if committed:
                mean_d = float(np.mean([r["delta_f1"] for r in committed]))
                mean_c = float(np.mean([r["commit_pct"] for r in committed]))
                print(f"  [{len(records)}/{N_QUESTIONS}] committed={len(committed)}  "
                      f"Δf1={mean_d:+.4f}  commit%={mean_c:.1f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            with open("/kaggle/working/exp_i_v2_partial.json", "w") as f:
                json.dump({"status": "partial", "n": len(records), "records": records}, f, indent=2)

    committed = [r for r in records if r["committed"]]
    uncommitted = [r for r in records if not r["committed"]]

    print(f"\n{'='*60}", flush=True)
    print(f"N committed={len(committed)}  N uncommitted={len(uncommitted)}", flush=True)

    if len(committed) < 20:
        print("Too few committed items for statistics", flush=True)
        return

    deltas = [r["delta_f1"] for r in committed]
    mean_delta = float(np.mean(deltas))
    std_delta  = float(np.std(deltas, ddof=1))
    se         = std_delta / np.sqrt(len(deltas))
    mean_cpct  = float(np.mean([r["commit_pct"] for r in committed]))
    frac_helped = sum(1 for d in deltas if d > 0.05) / len(deltas)
    frac_hurt   = sum(1 for d in deltas if d < -0.05) / len(deltas)

    # Parametric: one-sample t-test (H0: mean=0)
    t_stat, p_two  = stats.ttest_1samp(deltas, 0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    # Non-parametric: Wilcoxon signed-rank test
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(deltas, alternative="two-sided")
    except Exception:
        w_stat, p_wilcoxon = float("nan"), float("nan")

    # Bootstrap 95% CI
    ci_lo, ci_hi = bootstrap_ci(deltas)

    # Parametric CI
    ci_lo_t = mean_delta - 1.96 * se
    ci_hi_t = mean_delta + 1.96 * se

    print(f"Mean Δf1:          {mean_delta:+.5f} ± {std_delta:.5f} SD", flush=True)
    print(f"SE:                {se:.5f}", flush=True)
    print(f"95% CI (t):        [{ci_lo_t:+.5f}, {ci_hi_t:+.5f}]", flush=True)
    print(f"95% CI (boot):     [{ci_lo:+.5f}, {ci_hi:+.5f}]", flush=True)
    print(f"t-test:            t={t_stat:.3f}  p(two-tailed)={p_two:.4f}  p(one-tailed)={p_one:.4f}", flush=True)
    print(f"Wilcoxon:          W={w_stat}  p={p_wilcoxon:.4f}", flush=True)
    print(f"Helped (Δ>0.05):  {frac_helped:.2%}", flush=True)
    print(f"Hurt  (Δ<-0.05):  {frac_hurt:.2%}", flush=True)
    print(f"Mean commit%:      {mean_cpct:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)

    results = {
        "experiment": "EXP_I_EARLY_EXIT_CAUSAL_V2",
        "model": MODEL_ID,
        "probe_layer": LAYER_IDX,
        "n_total": len(records),
        "n_committed": len(committed),
        "n_uncommitted": len(uncommitted),
        "mean_delta_f1": round(mean_delta, 5),
        "std_delta_f1": round(std_delta, 5),
        "se": round(float(se), 5),
        "ci_95_t": [round(ci_lo_t, 5), round(ci_hi_t, 5)],
        "ci_95_boot": [round(ci_lo, 5), round(ci_hi, 5)],
        "t_stat": round(float(t_stat), 4),
        "p_two_tailed": round(float(p_two), 4),
        "p_one_tailed": round(float(p_one), 4),
        "wilcoxon_stat": round(float(w_stat), 2) if not np.isnan(w_stat) else None,
        "p_wilcoxon": round(float(p_wilcoxon), 4) if not np.isnan(p_wilcoxon) else None,
        "frac_helped": round(frac_helped, 4),
        "frac_hurt": round(frac_hurt, 4),
        "mean_commit_pct": round(mean_cpct, 2),
        "status": "complete"
    }
    with open("/kaggle/working/exp_i_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: /kaggle/working/exp_i_v2_results.json", flush=True)


if __name__ == "__main__":
    main()

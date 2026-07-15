"""
EXP-I: Early Exit Causal Truncation
=====================================
Causal complement to EXP-B's observational measurement.
EXP-B showed: Fisher probe detects commit point; most tokens post-commit.
EXP-I asks: if we ACTUALLY truncate at commit point (force </think>), what happens to F1?

Design: For each question, run TWICE:
  1. Full reasoning: complete think block, extract answer
  2. Truncated: detect commit point via Fisher probe, inject </think> at that step

Primary output: mean F1 delta (full - truncated) across ALL questions (not just diverging).
Distribution of per-question deltas reveals whether elaboration helps, hurts, or is neutral.

This converts EXP-B's observational f1_delta (+0.008, diverging cases only) to
a causal measurement across the full distribution.

Kill criterion: N/A (this is exploratory, not a theory-testing experiment)
"""

import gc, json, os, random, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
THINK_END_STR = "</think>"

POOL_SIZE        = 4000
N_QUESTIONS      = 200
LAYER_IDX        = 26
PCA_DIM          = 64
SEED             = 42
MAX_THINK_TOKENS = 600
MAX_ANS_TOKENS   = 80
COMMIT_WINDOW    = 5
COMMIT_PERSIST   = 3
CALIBRATION_N    = 40   # items for Fisher calibration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

rng = random.Random(SEED)
np.random.seed(SEED)


# ── Utilities ────────────────────────────────────────────────────────────────
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


def token_f1(pred: str, golds: list[str]) -> float:
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


def answer_contains(pred: str, golds: list[str]) -> bool:
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


# ── Hook helpers ─────────────────────────────────────────────────────────────
def extract_step1_hidden(mdl, tok, prompt: str, layer_idx: int) -> np.ndarray | None:
    hs_store = {}

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:
            hs_store["hs"] = hs[0, 0, :].float().cpu().numpy()

    h = mdl.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inp_t = tok(prompt, return_tensors="pt", truncation=True,
                    max_length=512).to(DEVICE)
        with torch.no_grad():
            _ = mdl.generate(**inp_t, max_new_tokens=1, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    finally:
        h.remove()
    return hs_store.get("hs")


# ── Calibration: build Fisher probe on PARAM items ───────────────────────────
def calibrate_probe(mdl, tok, pool, n_cal, layer_idx):
    """Use PARAM vs non-PARAM distinction for calibration (simpler than bilateral oracle)."""
    # Use pre-commit vs post-commit split using entropy: low entropy = committed
    # Collect hidden states from think-start tokens vs think-end tokens
    committed_hs = []
    uncommitted_hs = []

    t0 = time.time()
    for i, item in enumerate(pool[:n_cal * 10]):
        if len(committed_hs) >= n_cal and len(uncommitted_hs) >= n_cal:
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

        h = mdl.model.layers[layer_idx].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                pre = mdl(inp["input_ids"], use_cache=True)
                pkv = pre.past_key_values
            cur_ids = inp["input_ids"]
            steps_done = 0
            for step in range(100):  # short calibration run
                with torch.no_grad():
                    out = mdl(cur_ids[:, -1:], past_key_values=pkv, use_cache=True)
                logits = out.logits[0, -1, :]
                next_id = int(logits.argmax().item())
                pkv = out.past_key_values
                cur_ids = torch.tensor([[next_id]], device=DEVICE)
                cur_hs = hs_store.get("cur")
                if cur_hs is not None:
                    hs_list.append((step, cur_hs))
                if next_id == think_end_id or step >= 99:
                    break
                steps_done = step
        finally:
            h.remove()

        if len(hs_list) < 10:
            continue

        # Early steps = uncommitted, late steps = committed
        n_each = min(3, len(hs_list) // 3)
        for _, hs in hs_list[:n_each]:
            if len(uncommitted_hs) < n_cal:
                uncommitted_hs.append(hs)
        for _, hs in hs_list[-n_each:]:
            if len(committed_hs) < n_cal:
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

    # Threshold: median of committed scores
    com_scores = lda.decision_function(pca.transform(
        np.array(committed_hs[:n_min]).astype(np.float32)))
    thresh = float(np.percentile(com_scores, 20))
    print(f"Calibration done: n={n_min}/class  thresh={thresh:.4f}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    return pca, lda, thresh


# ── Full generation ───────────────────────────────────────────────────────────
def generate_full(mdl, tok, question: str) -> tuple[str, int, list[np.ndarray]]:
    """Full reasoning generation. Returns (answer, think_len, per-step hidden states)."""
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
        think_started = True
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
    """Run reasoning only until commit_step, then inject </think> and generate answer."""
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    think_end_id = tok.encode(THINK_END_STR, add_special_tokens=False)[-1]

    with torch.no_grad():
        pre = mdl(inp["input_ids"], use_cache=True)
        pkv = pre.past_key_values
    cur_ids = inp["input_ids"]

    # Run until commit_step
    for step in range(commit_step):
        with torch.no_grad():
            out = mdl(cur_ids[:, -1:], past_key_values=pkv, use_cache=True)
        next_id = int(out.logits[0, -1, :].argmax().item())
        pkv = out.past_key_values
        cur_ids = torch.tensor([[next_id]], device=DEVICE)

    # Force </think> — process it and sample first answer token from its logits
    think_ids = torch.tensor([[think_end_id]], device=DEVICE)
    with torch.no_grad():
        out = mdl(think_ids, past_key_values=pkv, use_cache=True)
    pkv = out.past_key_values
    first_ans_id = int(out.logits[0, -1, :].argmax().item())

    # Generate answer starting from first_ans_id (not re-processing </think>)
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
    print(f"EXP-I: Early Exit Causal Truncation", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"N_questions: {N_QUESTIONS}  Layer: {LAYER_IDX}", flush=True)

    hf_token = _get_hf_token()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True, token=hf_token
    ).to(DEVICE).eval()

    pool = load_pool(POOL_SIZE)

    # Calibrate probe
    print("\nPhase 1: Calibrating Fisher probe...", flush=True)
    pca, lda, thresh = calibrate_probe(mdl, tok, pool, CALIBRATION_N, LAYER_IDX)
    if pca is None:
        print("Calibration failed — aborting", flush=True)
        return

    # Main experiment: dual-run each question
    print("\nPhase 2: Dual-run experiment...", flush=True)
    records = []
    t0 = time.time()

    for i, item in enumerate(pool):
        if len(records) >= N_QUESTIONS:
            break
        q, ans = item["q"], item["answers"]

        # Full run
        try:
            answer_full, think_len, hs_traj = generate_full(mdl, tok, q)
        except Exception as e:
            continue

        if think_len == 0 or len(hs_traj) < COMMIT_WINDOW + COMMIT_PERSIST + 1:
            continue

        # Find commit point using Fisher probe
        hs_arr = np.array(hs_traj).astype(np.float32)
        X_pca = pca.transform(hs_arr)
        scores = lda.decision_function(X_pca).tolist()

        commit_step = None
        for s in range(COMMIT_WINDOW, len(scores) - COMMIT_PERSIST):
            if scores[s] > thresh:
                if all(scores[s+j] > thresh for j in range(COMMIT_PERSIST)):
                    commit_step = s
                    break

        if commit_step is None:
            # No commit detected — skip truncation, record as uncommitted
            f1_full = token_f1(answer_full, ans)
            records.append({
                "q": q, "commit_step": None,
                "think_len": think_len,
                "f1_full": f1_full, "f1_trunc": None,
                "delta_f1": None, "committed": False
            })
            continue

        # Truncated run at commit point
        try:
            answer_trunc = generate_truncated(mdl, tok, q, commit_step)
        except Exception as e:
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

        if (i+1) % 20 == 0 or len(records) % 20 == 0:
            committed = [r for r in records if r["committed"]]
            if committed:
                mean_delta = float(np.mean([r["delta_f1"] for r in committed]))
                mean_cpct  = float(np.mean([r["commit_pct"] for r in committed]))
                print(f"  [{len(records)}/{N_QUESTIONS}] committed={len(committed)}  "
                      f"mean_Δf1={mean_delta:+.4f}  mean_commit%={mean_cpct:.1f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            # Intermediate save
            with open("/kaggle/working/exp_i_partial.json", "w") as f:
                json.dump({"status": "partial", "records": records}, f, indent=2)

    committed = [r for r in records if r["committed"]]
    uncommitted = [r for r in records if not r["committed"]]

    if committed:
        deltas = [r["delta_f1"] for r in committed]
        mean_delta = float(np.mean(deltas))
        std_delta  = float(np.std(deltas))
        frac_helped = sum(1 for d in deltas if d > 0.05) / len(deltas)
        frac_hurt   = sum(1 for d in deltas if d < -0.05) / len(deltas)
        mean_cpct   = float(np.mean([r["commit_pct"] for r in committed]))
    else:
        mean_delta = std_delta = frac_helped = frac_hurt = mean_cpct = 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: n_committed={len(committed)} n_uncommitted={len(uncommitted)}", flush=True)
    print(f"Mean Δ_F1 (full-trunc): {mean_delta:+.4f} ± {std_delta:.4f}", flush=True)
    print(f"Helped (Δ>0.05): {frac_helped:.2%}  Hurt (Δ<-0.05): {frac_hurt:.2%}", flush=True)
    print(f"Mean commit%: {mean_cpct:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)

    results = {
        "experiment": "EXP_I_EARLY_EXIT_CAUSAL_V1",
        "model": MODEL_ID,
        "probe_layer": LAYER_IDX,
        "n_total": len(records),
        "n_committed": len(committed),
        "n_uncommitted": len(uncommitted),
        "mean_delta_f1": round(mean_delta, 5),
        "std_delta_f1": round(std_delta, 5),
        "frac_helped": round(frac_helped, 4),
        "frac_hurt": round(frac_hurt, 4),
        "mean_commit_pct": round(mean_cpct, 2),
        "status": "complete"
    }
    with open("/kaggle/working/exp_i_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: /kaggle/working/exp_i_results.json", flush=True)


if __name__ == "__main__":
    main()

"""
EXP-F: Commit-Point Hidden State Quality  (v2 — fast single-pass)
==========================================
At the detected commit token in a reasoning model's think block,
do hidden states predict whether the final answer will be CC or CW?

Scientific question (Q1/L3): Is early commitment "informed" or "blind"?
  - AUROC > 0.70 at commit point -> commit is INFORMED (Gate 3 viable)
  - AUROC ~ 0.50 at commit point -> commit is BLIND (model commits blind)

v2 changes vs v1:
  - Hidden state captured DURING Phase 1 generation (no Phase 2 double-pass)
  - MAX_THINK_TOKENS reduced 1024 -> 300  (~3x faster per item)
  - N_CC/CW_TARGET reduced 80 -> 40  (~2x fewer items)
  - Removed broken entropy pre-filter (THETA_CONF was calibrated on base model
    answer tokens, not reasoning model think tokens — let everything through)
  - Print progress every 10 items
  - Intermediate saves
"""

import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID         = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
THINK_END_STR    = "</think>"

POOL_SIZE        = 3000
N_CC_TARGET      = 40
N_CW_TARGET      = 40
LAYER_IDX        = 26
PCA_DIM          = 64
SEED             = 42
MAX_THINK_TOKENS = 300    # v2: reduced from 1024
MAX_ANS_TOKENS   = 64
COMMIT_WINDOW    = 5
COMMIT_PERSIST   = 3
TRAIN_FRAC       = 0.75
MIN_ITEMS        = 20     # abort if below this per class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

rng = random.Random(SEED)
np.random.seed(SEED)


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


def token_f1(pred, golds):
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


def answer_contains(pred, golds):
    return any(g.lower() in pred.lower() for g in golds)


def fmt_prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def load_pool(n):
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


def output_entropy(logits):
    probs = torch.softmax(logits.float(), dim=-1).clamp(min=1e-10)
    return float(-torch.sum(probs * torch.log(probs)).item())


def find_commit(traj, window=COMMIT_WINDOW, persist=COMMIT_PERSIST, thresh=0.0):
    """Detect commit point in an inverted-entropy trajectory (high = committed)."""
    for i in range(window, len(traj) - persist):
        if traj[i] > thresh and all(traj[i+j] > thresh for j in range(persist)):
            return i
    return None


def generate_and_collect(mdl, tok, question, probe_layer):
    """
    Single-pass: generate think block, detect commit, return hidden state at
    commit step (captured inline — no second pass needed).
    """
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    think_end_id = tok.encode(THINK_END_STR, add_special_tokens=False)[0]

    hs_store = {}

    def hook_fn(module, inp_h, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:
            hs_store["cur"] = hs[0, 0, :].float().cpu().numpy().copy()

    h = mdl.model.layers[probe_layer].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            pre = mdl(inp["input_ids"], use_cache=True)
            pkv = pre.past_key_values
        cur_ids = inp["input_ids"][:, -1:]

        think_entropies = []
        hs_per_step = []   # store hs at each think step

        # Generate think block
        for step in range(MAX_THINK_TOKENS):
            with torch.no_grad():
                out = mdl(cur_ids, past_key_values=pkv, use_cache=True)
            logits = out.logits[0, -1, :]
            next_id = int(logits.argmax().item())
            pkv = out.past_key_values
            cur_ids = torch.tensor([[next_id]], device=DEVICE)

            think_entropies.append(output_entropy(logits))
            hs_per_step.append(hs_store.get("cur", None))

            if next_id == think_end_id:
                break

        # Find commit from inverted entropy (low entropy = committed = high inverted)
        inverted = [-e for e in think_entropies]
        # Normalize by local mean to get relative threshold
        if len(inverted) > COMMIT_WINDOW + COMMIT_PERSIST + 1:
            mean_inv = float(np.mean(inverted))
            commit_step = find_commit(inverted, thresh=mean_inv)
        else:
            commit_step = None

        # Generate answer
        ans_tokens = []
        for _ in range(MAX_ANS_TOKENS):
            with torch.no_grad():
                out = mdl(cur_ids, past_key_values=pkv, use_cache=True)
            logits = out.logits[0, -1, :]
            next_id = int(logits.argmax().item())
            pkv = out.past_key_values
            cur_ids = torch.tensor([[next_id]], device=DEVICE)
            ans_tokens.append(next_id)
            if next_id == tok.eos_token_id:
                break

        answer = tok.decode(ans_tokens, skip_special_tokens=True).strip()

        # Get commit-point hidden state from the stored list
        commit_hs = None
        if commit_step is not None and commit_step < len(hs_per_step):
            commit_hs = hs_per_step[commit_step]

    finally:
        h.remove()

    return {
        "answer":      answer,
        "think_len":   len(think_entropies),
        "commit_step": commit_step,
        "commit_hs":   commit_hs,
    }


def pca_lda_auroc(X_tr, y_tr, X_te, y_te):
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca.transform(X_te.astype(np.float32))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_r, y_tr)
    scores = lda.decision_function(X_te_r)
    auroc = float(roc_auc_score(y_te, scores))
    y_shuf = y_tr.copy(); np.random.shuffle(y_shuf)
    lda2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda2.fit(X_tr_r, y_shuf)
    shuf = float(roc_auc_score(y_te, lda2.decision_function(X_te_r)))
    return round(auroc, 4), round(shuf, 4)


def main():
    print("EXP-F v2: Commit-Point Hidden State Quality (single-pass)", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Layer: {LAYER_IDX}  PCA: {PCA_DIM}  MAX_THINK: {MAX_THINK_TOKENS}", flush=True)
    print(f"N_target: {N_CC_TARGET} CC + {N_CW_TARGET} CW", flush=True)

    hf_token = _get_hf_token()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True, token=hf_token
    ).to(DEVICE).eval()

    pool = load_pool(POOL_SIZE)
    print(f"Pool loaded: {len(pool)} questions", flush=True)

    cc_items, cw_items = [], []
    t0 = time.time()
    n_no_commit = 0
    n_processed = 0

    print("\nCollecting CC/CW items with commit-point hidden states...", flush=True)

    for i, item in enumerate(pool):
        if len(cc_items) >= N_CC_TARGET and len(cw_items) >= N_CW_TARGET:
            break

        q, ans = item["q"], item["answers"]
        result = generate_and_collect(mdl, tok, q, LAYER_IDX)
        n_processed += 1

        if result["commit_step"] is None or result["commit_hs"] is None:
            n_no_commit += 1
            continue

        f1 = token_f1(result["answer"], ans)
        is_correct = f1 >= 0.30 or answer_contains(result["answer"], ans)

        record = {
            "q": q, "answers": ans,
            "commit_step": result["commit_step"],
            "think_len":   result["think_len"],
            "answer":      result["answer"],
            "f1":          f1,
            "commit_hs":   result["commit_hs"].tolist(),
        }

        if is_correct and len(cc_items) < N_CC_TARGET:
            cc_items.append(record)
        elif not is_correct and len(cw_items) < N_CW_TARGET:
            cw_items.append(record)

        if (n_processed) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_processed / elapsed
            print(f"  [{n_processed}] CC={len(cc_items)} CW={len(cw_items)}  "
                  f"no_commit={n_no_commit}  {rate:.2f}q/s  {elapsed:.0f}s", flush=True)

        # Intermediate save every 20 collected items
        if (len(cc_items) + len(cw_items)) % 20 == 0 and (len(cc_items) + len(cw_items)) > 0:
            inter = {
                "status": "collecting",
                "cc_so_far": len(cc_items),
                "cw_so_far": len(cw_items),
                "n_processed": n_processed,
                "elapsed_s": round(time.time() - t0, 1),
            }
            with open("/kaggle/working/exp_f_results.json", "w") as f:
                json.dump(inter, f)
            print(f"  [intermediate save] CC={len(cc_items)} CW={len(cw_items)}", flush=True)

    elapsed_collect = time.time() - t0
    print(f"\nCollection done: CC={len(cc_items)} CW={len(cw_items)} "
          f"in {elapsed_collect:.0f}s  ({n_processed} items scanned, "
          f"{n_no_commit} no commit)", flush=True)

    if len(cc_items) < MIN_ITEMS or len(cw_items) < MIN_ITEMS:
        msg = f"INSUFFICIENT ITEMS: CC={len(cc_items)} CW={len(cw_items)} (min={MIN_ITEMS})"
        print(msg, flush=True)
        with open("/kaggle/working/exp_f_results.json", "w") as f:
            json.dump({"status": "aborted", "reason": msg}, f)
        return

    # Build hidden state arrays
    n_min = min(len(cc_items), len(cw_items))
    cc_hs = np.array([item["commit_hs"] for item in cc_items[:n_min]])
    cw_hs = np.array([item["commit_hs"] for item in cw_items[:n_min]])
    X = np.vstack([cc_hs, cw_hs])
    y = np.array([1] * n_min + [0] * n_min)
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    n_tr = int(len(y) * TRAIN_FRAC)

    print(f"\nFitting Fisher+PCA64  N={n_min}/class  n_train={n_tr}  n_test={len(y)-n_tr}", flush=True)
    auroc, shuf_auroc = pca_lda_auroc(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])

    verdict_flag = "CLEAN" if shuf_auroc < auroc - 0.05 else "WARN"
    verdict = "INFORMED" if auroc >= 0.70 else "BLIND"

    print(f"\n{'='*60}", flush=True)
    print(f"COMMIT-POINT AUROC: {auroc:.4f}  shuffled={shuf_auroc:.4f} ({verdict_flag})", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"  >= 0.70 = INFORMED (commit point encodes answer quality -> Gate 3 viable)", flush=True)
    print(f"  < 0.70  = BLIND (model commits before quality is determined)", flush=True)
    print(f"{'='*60}", flush=True)

    # Commit stats
    cc_commit_pcts = [100.0 * (it["think_len"] - it["commit_step"]) / it["think_len"]
                      for it in cc_items[:n_min]]
    cw_commit_pcts = [100.0 * (it["think_len"] - it["commit_step"]) / it["think_len"]
                      for it in cw_items[:n_min]]
    print(f"CC mean commit_pct={np.mean(cc_commit_pcts):.1f}%  "
          f"CW mean commit_pct={np.mean(cw_commit_pcts):.1f}%", flush=True)

    results = {
        "experiment": "EXP_F_COMMIT_QUALITY_V1",
        "version": "v2",
        "model": MODEL_ID,
        "probe_layer": LAYER_IDX,
        "commit_point_auroc": auroc,
        "shuf_auroc": shuf_auroc,
        "shuffled_flag": verdict_flag,
        "verdict": verdict,
        "n_per_class": n_min,
        "n_train": n_tr,
        "n_test": len(y) - n_tr,
        "n_processed": n_processed,
        "n_no_commit": n_no_commit,
        "cc_mean_commit_pct": round(float(np.mean(cc_commit_pcts)), 2),
        "cw_mean_commit_pct": round(float(np.mean(cw_commit_pcts)), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "status": "complete",
    }
    with open("/kaggle/working/exp_f_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: /kaggle/working/exp_f_results.json", flush=True)


if __name__ == "__main__":
    main()

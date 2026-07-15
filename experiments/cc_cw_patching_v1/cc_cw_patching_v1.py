"""
EXP-H: CC/CW Causal Patching
==============================
Tests whether patching CW items toward the CC centroid direction at L26
step-0 produces F1 gains.

Background: C005 showed centroid patching is epiphenomenal for PARAM/CTX_DEP.
CC/CW Fisher gap=0.240 is ~4x larger than PARAM/CTX_DEP gap — more geometric
room for the patch to work.

Scientific question (Q2): Is the CC/CW Fisher geometry causal?
  - F1 gain > 0.05 at any lambda -> causal leverage confirmed (significant finding)
  - All lambdas null -> CC/CW geometry is also epiphenomenal at centroid level

Kill criterion: Null patching at all magnitudes on CC/CW ->
  geometry is epiphenomenal for confabulation geometry too.
  Stop pursuing causal patching at centroid level in this program.
"""

import gc, json, os, random, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

POOL_SIZE    = 5000
N_CC_TARGET  = 80
N_CW_TARGET  = 80
THETA_CONF   = 1.1043
LAYER_IDX    = 26
PCA_DIM      = 64
SEED         = 42
MAX_NEW      = 60
TRAIN_FRAC   = 0.75

PARAM_MIN_F1 = 0.50
CTX_MAX_NC   = 0.05
CTX_MIN_CTX  = 0.50

LAMBDAS = [0.0, 0.25, 0.50, 1.0, 2.0, 4.0]   # patch magnitude scale factors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

rng = random.Random(SEED)
np.random.seed(SEED)


# ── Shared utilities ──────────────────────────────────────────────────────────
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


def fmt_prompt(q: str, ctx: str = "") -> str:
    if ctx:
        return f"Context: {ctx}\n\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


def output_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1).clamp(min=1e-10)
    return float(-torch.sum(probs * torch.log(probs)).item())


def load_pool(n: int):
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
    pool = []
    for ex in ds:
        if len(pool) >= n:
            break
        q   = ex["question"]
        ans = ex["answer"]["aliases"][:5]
        ctx = (ex.get("search_results", {}).get("search_context") or [""])
        ctx = ctx[0][:800] if ctx else ""
        if q and ans:
            pool.append({"q": q, "answers": ans, "ctx": ctx})
    rng.shuffle(pool)
    return pool


# ── Hidden state extraction (prefill = step-0) ───────────────────────────────
def extract_prefill_hidden(mdl, tok, prompt: str, layer_idx: int) -> np.ndarray | None:
    hs_store = {}

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        # Capture last token of prefill
        hs_store["hs"] = hs[0, -1, :].float().cpu().numpy()

    h = mdl.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inp_t = tok(prompt, return_tensors="pt", truncation=True,
                    max_length=512).to(DEVICE)
        with torch.no_grad():
            _ = mdl(**inp_t)
    finally:
        h.remove()
    return hs_store.get("hs")


# ── Patched generation ────────────────────────────────────────────────────────
def gen_with_patch(mdl, tok, prompt: str, patch_vec: np.ndarray,
                   layer_idx: int, lam: float) -> str:
    """Generate with additive patch at step-0 (first generated token)."""
    patch_t = torch.from_numpy(patch_vec.astype(np.float32))
    patched = {"done": False}

    def hook_fn(module, inp, out):
        if patched["done"]:
            return
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:   # generation step
            patched["done"] = True
            p = patch_t.to(hs.device).to(hs.dtype)
            hs_mod = hs + lam * p.unsqueeze(0).unsqueeze(0)
            if isinstance(out, tuple):
                return (hs_mod,) + out[1:]
            return hs_mod

    h = mdl.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inp_t = tok(prompt, return_tensors="pt", truncation=True,
                    max_length=512).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(
                **inp_t, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.eos_token_id, use_cache=True
            )
        return tok.decode(out[0][inp_t["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
    finally:
        h.remove()


# ── Bilateral oracle collection with entropy filter ───────────────────────────
def collect_cc_cw(mdl, tok, pool, n_target):
    """Collect CC and CW items (both near THETA_CONF entropy) from PARAM questions."""
    cc_items, cw_items = [], []
    t0 = time.time()

    for i, item in enumerate(pool):
        if len(cc_items) >= n_target and len(cw_items) >= n_target:
            break
        q, ans = item["q"], item["answers"]
        prompt = fmt_prompt(q)
        inp = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=512).to(DEVICE)
        with torch.no_grad():
            out = mdl(**inp)
        ent = output_entropy(out.logits[0, -1, :])

        if abs(ent - THETA_CONF) > 0.30:
            continue

        gen = tok.decode(
            mdl.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                          pad_token_id=tok.eos_token_id)[0][inp["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        f1 = token_f1(gen, ans)
        is_correct = f1 >= 0.30 or answer_contains(gen, ans)

        record = {"q": q, "answers": ans, "answer": gen, "f1": f1, "entropy": ent}
        if is_correct and len(cc_items) < n_target:
            cc_items.append(record)
        elif not is_correct and len(cw_items) < n_target:
            cw_items.append(record)

        if (i+1) % 50 == 0:
            print(f"  [{i+1}] CC={len(cc_items)} CW={len(cw_items)}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    return cc_items, cw_items


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"EXP-H: CC/CW Causal Patching", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Layer: {LAYER_IDX}  Lambdas: {LAMBDAS}", flush=True)

    hf_token = _get_hf_token()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True, token=hf_token
    ).to(DEVICE).eval()

    pool = load_pool(POOL_SIZE)

    # Collect CC/CW items
    print("\nPhase 1: Collecting CC/CW items...", flush=True)
    cc_items, cw_items = collect_cc_cw(mdl, tok, pool, N_CC_TARGET)
    print(f"Collected: CC={len(cc_items)} CW={len(cw_items)}", flush=True)

    if len(cc_items) < 20 or len(cw_items) < 20:
        print("INSUFFICIENT — aborting", flush=True)
        return

    # Extract prefill hidden states for Fisher calibration
    print("\nPhase 2: Extracting hidden states...", flush=True)
    cc_hs, cw_hs = [], []
    for item in cc_items:
        hs = extract_prefill_hidden(mdl, tok, fmt_prompt(item["q"]), LAYER_IDX)
        if hs is not None:
            cc_hs.append(hs)
    for item in cw_items:
        hs = extract_prefill_hidden(mdl, tok, fmt_prompt(item["q"]), LAYER_IDX)
        if hs is not None:
            cw_hs.append(hs)

    # Compute CC centroid and Fisher direction
    n_min = min(len(cc_hs), len(cw_hs))
    X_cc = np.array(cc_hs[:n_min])
    X_cw = np.array(cw_hs[:n_min])
    cc_centroid = X_cc.mean(axis=0)
    cw_centroid = X_cw.mean(axis=0)

    # PCA + LDA for Fisher axis
    X_all = np.vstack([X_cc, X_cw])
    y_all = np.array([1] * n_min + [0] * n_min)
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_pca = pca.fit_transform(X_all.astype(np.float32))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_pca, y_all)

    # Baseline AUROC
    scores = lda.decision_function(X_pca)
    baseline_auroc = float(roc_auc_score(y_all, scores))
    print(f"Baseline Fisher AUROC (train): {baseline_auroc:.4f}", flush=True)

    # Patch direction: mean(CC) - mean(CW) in original space
    patch_direction = (cc_centroid - cw_centroid).astype(np.float32)
    patch_direction /= (np.linalg.norm(patch_direction) + 1e-10)

    # Phase 3: Patching experiments on CW items
    print("\nPhase 3: Patching CW items toward CC centroid...", flush=True)
    lambda_results = {}
    t0 = time.time()

    for lam in LAMBDAS:
        f1_scores = []
        for j, item in enumerate(cw_items[:n_min]):
            patched_ans = gen_with_patch(
                mdl, tok, fmt_prompt(item["q"]),
                patch_direction, LAYER_IDX, lam
            )
            f1 = token_f1(patched_ans, item["answers"])
            f1_scores.append(f1)

        mean_f1 = float(np.mean(f1_scores))
        orig_f1 = float(np.mean([i["f1"] for i in cw_items[:n_min]]))
        delta = mean_f1 - orig_f1
        lambda_results[str(lam)] = {
            "lambda": lam, "mean_f1": mean_f1,
            "orig_f1": orig_f1, "delta_f1": delta
        }
        print(f"  lambda={lam:.2f}  F1={mean_f1:.4f}  "
              f"orig={orig_f1:.4f}  Δ={delta:+.4f}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    # Verdict
    max_delta = max(r["delta_f1"] for r in lambda_results.values())
    verdict = "CAUSAL_LEVERAGE" if max_delta > 0.05 else "EPIPHENOMENAL"

    print(f"\n{'='*60}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Max Δ_F1 = {max_delta:+.4f} (> 0.05 = causal leverage)", flush=True)
    print(f"{'='*60}", flush=True)

    results = {
        "experiment": "EXP_H_CC_CW_PATCHING_V1",
        "model": MODEL_ID,
        "probe_layer": LAYER_IDX,
        "lambdas_tested": LAMBDAS,
        "baseline_fisher_auroc": baseline_auroc,
        "n_cc": len(cc_hs), "n_cw": len(cw_hs),
        "lambda_results": lambda_results,
        "max_delta_f1": max_delta,
        "verdict": verdict,
        "status": "complete"
    }
    with open("/kaggle/working/exp_h_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: /kaggle/working/exp_h_results.json", flush=True)


if __name__ == "__main__":
    main()

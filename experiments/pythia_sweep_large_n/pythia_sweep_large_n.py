"""
EXP-K: Large-N Pythia Checkpoint Sweep
========================================
Measures bilateral oracle Fisher+PCA64 AUROC at N>=50/class per checkpoint
across 8 Pythia-1.4b training checkpoints.

Scientific question (Q3): How does epistemic legibility emerge during training?
Does AUROC follow INVERTED_U (prior provisional result) or MONOTONE growth?

Claims tested: C011 (INVERTED_U vs MONOTONE vs FLAT)
Kill criterion: FLAT (variance < 0.05 across all checkpoints) ->
               legibility is architecturally determined, not learned.
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
CHECKPOINTS = [
    "EleutherAI/pythia-1.4b",                          # latest (step143000)
    "EleutherAI/pythia-1.4b-deduped",                  # deduped variant
]
# Use revision=stepXXXX to load specific checkpoints.
# step512–step8000 excluded: Pythia-1.4b at those stages has 0% bilateral oracle
# yield on TriviaQA (model has not yet developed sufficient factual recall).
# First capable checkpoint empirically observed around step16000.
REVISIONS = [
    "step16000",
    "step33000",
    "step66000",
    "step143000",
]
BASE_MODEL = "EleutherAI/pythia-1.4b"   # non-deduped for checkpoint revisions

POOL_SIZE   = 8000
N_TARGET    = 60      # per class (floor; bump to 60+ for clean CIs)
N_CALIB     = 45      # per class for calibration
LAYER_IDX   = 17      # Pythia-1.4b has 24 layers; L17 ≈ 70% depth
PCA_DIM     = 64
SEED        = 42
MAX_NEW     = 60
PARAM_MIN_F1    = 0.50
CTX_MAX_NC      = 0.05
CTX_MIN_CTX     = 0.50
TRAIN_FRAC  = 0.75

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
    pl = pred.lower()
    return any(g.lower() in pl for g in golds)


def fmt_prompt(q: str, ctx: str = "") -> str:
    if ctx:
        return f"Context: {ctx}\n\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


def gen_text(mdl, tok, prompt: str, max_new: int = MAX_NEW) -> str:
    inp = tok(prompt, return_tensors="pt", truncation=True,
              max_length=512).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(
            **inp, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


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


# ── Architecture-agnostic layer accessor ─────────────────────────────────────
def get_layer(mdl, layer_idx: int):
    """Return the transformer layer regardless of model family."""
    if hasattr(mdl, 'model') and hasattr(mdl.model, 'layers'):
        return mdl.model.layers[layer_idx]          # Qwen, Llama, Gemma, Mistral
    elif hasattr(mdl, 'gpt_neox') and hasattr(mdl.gpt_neox, 'layers'):
        return mdl.gpt_neox.layers[layer_idx]       # Pythia (GPTNeoX)
    elif hasattr(mdl, 'transformer') and hasattr(mdl.transformer, 'h'):
        return mdl.transformer.h[layer_idx]         # GPT-2 style
    else:
        raise ValueError(f"Unknown architecture: {type(mdl).__name__}")


# ── Explicit 2-step hidden state extraction (architecture-agnostic) ───────────
def extract_step1_hidden(mdl, tok, prompt: str, layer_idx: int) -> np.ndarray | None:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    try:
        with torch.no_grad():
            prefill = mdl(
                **inp,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            next_tok = prefill.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            gen = mdl(
                input_ids=next_tok,
                past_key_values=prefill.past_key_values,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states[0]=embeddings, [k]=output of layer k-1 (0-indexed)
        return gen.hidden_states[layer_idx + 1][0, 0, :].detach().float().cpu().numpy()
    except Exception as e:
        print(f"  WARN: extract_step1_hidden failed: {e}", flush=True)
        return None


# ── Bilateral Oracle ──────────────────────────────────────────────────────────
def label_item(mdl, tok, item: dict) -> tuple[str, np.ndarray | None]:
    q, ans, ctx = item["q"], item["answers"], item["ctx"]
    nc = gen_text(mdl, tok, fmt_prompt(q))
    nc_f1 = token_f1(nc, ans)

    if nc_f1 >= PARAM_MIN_F1 or answer_contains(nc, ans):
        hs = extract_step1_hidden(mdl, tok, fmt_prompt(q), LAYER_IDX)
        return "PARAM", hs

    if nc_f1 <= CTX_MAX_NC and ctx:
        wc = gen_text(mdl, tok, fmt_prompt(q, ctx))
        if token_f1(wc, ans) >= CTX_MIN_CTX or answer_contains(wc, ans):
            hs = extract_step1_hidden(mdl, tok, fmt_prompt(q), LAYER_IDX)
            return "CTX_DEP", hs

    return "SKIP", None


# ── Fisher+PCA64 ──────────────────────────────────────────────────────────────
def pca_lda_auroc(X_tr, y_tr, X_te, y_te):
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_tr_r = pca.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca.transform(X_te.astype(np.float32))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_r, y_tr)
    scores = lda.decision_function(X_te_r)
    auroc = float(roc_auc_score(y_te, scores))

    # shuffled control
    y_shuf = y_tr.copy()
    np.random.shuffle(y_shuf)
    lda2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda2.fit(X_tr_r, y_shuf)
    shuf_auroc = float(roc_auc_score(y_te, lda2.decision_function(X_te_r)))
    return round(auroc, 4), round(shuf_auroc, 4)


EARLY_ABORT_AFTER = 2000  # if still 0/0 after this many items, skip checkpoint

def collect_items(mdl, tok, pool, n_target):
    param_hs, ctx_hs = [], []
    t0 = time.time()
    for i, item in enumerate(pool):
        if len(param_hs) >= n_target and len(ctx_hs) >= n_target:
            break
        # Early abort: if both yields are zero after EARLY_ABORT_AFTER items,
        # this checkpoint cannot support the bilateral oracle — skip it.
        if i >= EARLY_ABORT_AFTER and len(param_hs) == 0 and len(ctx_hs) == 0:
            print(f"  EARLY_ABORT at [{i+1}]: zero yield — bilateral oracle "
                  f"not applicable at this checkpoint", flush=True)
            break
        label, hs = label_item(mdl, tok, item)
        if label == "PARAM" and len(param_hs) < n_target and hs is not None:
            param_hs.append(hs)
        elif label == "CTX_DEP" and len(ctx_hs) < n_target and hs is not None:
            ctx_hs.append(hs)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}] PARAM={len(param_hs)} CTX_DEP={len(ctx_hs)}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return param_hs, ctx_hs


def run_checkpoint(revision: str, pool: list) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"CHECKPOINT: {revision}", flush=True)
    print(f"{'='*60}", flush=True)

    hf_token = _get_hf_token()
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=revision,
        trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=revision,
        torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True, token=hf_token
    ).to(DEVICE).eval()

    param_hs, ctx_hs = collect_items(mdl, tok, pool, N_TARGET)

    result = {"revision": revision, "n_param": len(param_hs),
              "n_ctx_dep": len(ctx_hs), "auroc": None,
              "shuf_auroc": None, "status": "insufficient"}

    n_min = min(len(param_hs), len(ctx_hs))
    if n_min >= 20:
        X = np.array(param_hs[:n_min] + ctx_hs[:n_min])
        y = np.array([1] * n_min + [0] * n_min)
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        n_tr = int(len(y) * TRAIN_FRAC)
        auroc, shuf = pca_lda_auroc(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
        result.update({"auroc": auroc, "shuf_auroc": shuf,
                       "n_test": len(y) - n_tr, "status": "complete"})
        verdict = "CLEAN" if shuf < auroc - 0.05 else "WARN"
        print(f"  AUROC={auroc:.4f}  shuffled={shuf:.4f} ({verdict})  "
              f"n_min={n_min}  elapsed={time.time()-t0:.0f}s", flush=True)
    else:
        print(f"  INSUFFICIENT: n_param={len(param_hs)} n_ctx={len(ctx_hs)}", flush=True)

    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    return result


def determine_shape(results: list[dict]) -> str:
    valid = [(r["revision"], r["auroc"]) for r in results
             if r["auroc"] is not None]
    if len(valid) < 4:
        return "INSUFFICIENT_DATA"
    aurocs = [a for _, a in valid]
    peak_i = int(np.argmax(aurocs))
    end_val = aurocs[-1]
    start_val = aurocs[0]
    peak_val = aurocs[peak_i]
    variance = float(np.var(aurocs))
    if variance < 0.002:
        return "FLAT"
    if peak_i > 0 and peak_i < len(aurocs) - 1 and peak_val - end_val > 0.05:
        return "INVERTED_U"
    if aurocs[-1] > aurocs[0] + 0.05:
        return "MONOTONE_RISE"
    return "NOISY"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"EXP-K: Large-N Pythia Checkpoint Sweep", flush=True)
    print(f"Model: {BASE_MODEL}", flush=True)
    print(f"Revisions: {REVISIONS}", flush=True)
    print(f"N_TARGET={N_TARGET}/class  PCA_DIM={PCA_DIM}  LAYER={LAYER_IDX}", flush=True)

    print("\nLoading pool...", flush=True)
    pool = load_pool(POOL_SIZE)
    print(f"Pool loaded: {len(pool)} items", flush=True)

    all_results = []
    for rev in REVISIONS:
        try:
            r = run_checkpoint(rev, pool)
            all_results.append(r)
            # Intermediate save
            with open("/kaggle/working/exp_k_partial.json", "w") as f:
                json.dump({"status": "partial", "results": all_results}, f, indent=2)
        except Exception as e:
            print(f"  ERROR at {rev}: {e}", flush=True)
            all_results.append({"revision": rev, "status": "error", "error": str(e)})

    shape = determine_shape(all_results)

    print(f"\n{'='*60}", flush=True)
    print(f"SHAPE VERDICT: {shape}", flush=True)
    print(f"{'='*60}", flush=True)
    for r in all_results:
        if r.get("auroc") is not None:
            print(f"  {r['revision']:15s}  AUROC={r['auroc']:.4f}  "
                  f"shuf={r['shuf_auroc']:.4f}  "
                  f"n={r.get('n_param',0)}/{r.get('n_ctx_dep',0)}", flush=True)

    final = {
        "experiment": "EXP_K_PYTHIA_LARGE_N_V4",
        "model": BASE_MODEL,
        "layer_idx": LAYER_IDX,
        "n_target": N_TARGET,
        "shape_verdict": shape,
        "results": all_results,
        "status": "complete"
    }
    with open("/kaggle/working/exp_k_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print("\nSaved: /kaggle/working/exp_k_results.json", flush=True)


if __name__ == "__main__":
    main()

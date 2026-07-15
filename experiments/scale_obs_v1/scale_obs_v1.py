#!/usr/bin/env python3
"""
scale_obs_v1.py — EXP-E: Observability Scaling Science

SCIENTIFIC QUESTION:
  Does bilateral oracle AUROC scale with parameter count?
  Is Fisher+PCA64 AUROC a monotone function of log10(params)?
  Does entropy AUROC track the same way?

CONTEXT:
  The observability ladder requires knowing whether the signal quality
  we measure is architecture-specific (e.g. Qwen-family only), or
  whether it follows a scaling law. If AUROC ~ f(log10(params)), then
  observability is a function of model scale. If AUROC is flat or
  non-monotone, size doesn't predict observability.

  EXP-E is an invariance test (Principle P7): does the measurement
  remain consistent across architectures and scale regimes?

DESIGN:
  5 models run serially (load → run → unload):
    Qwen2.5-0.5B-Instruct   ~500M
    Qwen2.5-1.5B-Instruct   ~1.5B
    Qwen2.5-3B-Instruct     ~3B
    Llama-3.2-1B-Instruct   ~1B
    Llama-3.2-3B-Instruct   ~3B

  Per model:
    - Bilateral oracle collection (N_TARGET=80 per class: PARAM, CTX_DEP)
    - Fisher+PCA64 AUROC at penultimate layer, step-1
    - Output entropy AUROC at step-1
    - Record: n_params, fisher_auroc, entropy_auroc, n_layers, theta_conf

  Analysis:
    - Spearman correlation: AUROC vs log10(params) for Fisher and entropy
    - Fisher/Entropy gap vs log10(params)
    - Is AUROC monotone in scale?
    - Family comparison: Qwen vs Llama at matched scale

DECISION GATE:
  MONOTONE_FISHER   : Spearman ρ(Fisher, log_params) ≥ 0.80
  MONOTONE_ENTROPY  : Spearman ρ(entropy, log_params) ≥ 0.80
  FLAT_BOTH         : both ρ ≤ 0.40 — size predicts neither
  FAMILY_DIVERGENCE : Llama and Qwen differ ≥ 0.10 AUROC at matched scale
  INCONCLUSIVE      : N < 4 models complete

GPU: T4. Expected ~3-5h (serial model loading).
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "qwen25_0.5b",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "qwen",
        "n_params": 0.5e9,
    },
    {
        "name": "qwen25_1.5b",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "qwen",
        "n_params": 1.5e9,
    },
    {
        "name": "qwen25_3b",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "family": "qwen",
        "n_params": 3.0e9,
    },
    {
        "name": "llama32_1b",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "llama",
        "n_params": 1.0e9,
    },
    {
        "name": "llama32_3b",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama",
        "n_params": 3.0e9,
    },
]

POOL_SIZE    = 3_000     # per model (keep small for T4 serial run)
N_TARGET     = 80        # per class
PARAM_MIN_F1 = 0.50
CTX_MAX_NC   = 0.05
CTX_MIN_CTX  = 0.50
TRAIN_FRAC   = 0.75
ENTROPY_PCT  = 50        # use median as oracle split, not confidence zone
PCA_DIM      = 64
SEED         = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── HF Token ──────────────────────────────────────────────────────────────────
def _get_hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret("HF_TOKEN")
        if v:
            return v
    except Exception:
        pass
    return None


# ── Data ───────────────────────────────────────────────────────────────────────
def load_pool(n: int):
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      trust_remote_code=True)
    items = []
    for ex in ds:
        ep = ex.get("entity_pages", {})
        ctx = (ep.get("wiki_context") or [""])[0][:800] if ep else ""
        ans = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        items.append({"question": ex["question"], "context": ctx, "answers": ans})
        if len(items) >= n:
            break
    random.shuffle(items)
    return items


def token_f1(pred: str, golds: list) -> float:
    pt = set(pred.lower().split())
    best = 0.0
    for g in golds:
        gt = set(g.lower().split())
        if not pt or not gt:
            continue
        c = pt & gt
        if not c:
            continue
        p = len(c) / len(pt)
        r = len(c) / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def fmt_nocontext(q: str) -> str:
    return f"Answer in one short phrase.\nQuestion: {q}\nAnswer:"


def fmt_withcontext(q: str, ctx: str) -> str:
    return (f"Use the passage to answer.\n"
            f"Passage: {ctx}\nQuestion: {q}\nAnswer:")


# ── Per-model run ─────────────────────────────────────────────────────────────
def run_model(cfg: dict, pool: list) -> dict | None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_hf = _get_hf_token()
    if tok_hf:
        from huggingface_hub import login
        login(token=tok_hf, add_to_git_credential=False)

    print(f"\n{'='*60}", flush=True)
    print(f"Loading {cfg['model_id']}", flush=True)
    t_load = time.time()

    try:
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        mdl = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16,
            device_map=None, trust_remote_code=True
        ).to(DEVICE).eval()
    except Exception as e:
        print(f"  LOAD FAILED: {e}", flush=True)
        return None

    n_layers  = mdl.config.num_hidden_layers
    layer_idx = n_layers - 2  # penultimate
    print(f"  n_layers={n_layers}, penultimate={layer_idx}", flush=True)
    print(f"  Load time: {time.time()-t_load:.1f}s", flush=True)

    def extract_step1(question: str, context: str | None):
        """Single forward pass, return (hidden_state, entropy)."""
        if context:
            prompt = fmt_withcontext(question, context)
        else:
            prompt = fmt_nocontext(question)

        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            try:
                out = mdl.generate(
                    **inp,
                    max_new_tokens=1,
                    do_sample=False,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tok.eos_token_id,
                    use_cache=True,
                )
                hs  = out.hidden_states[0][layer_idx][0, -1, :].float().cpu().numpy()
                logits = out.scores[0][0].float()
                probs  = torch.softmax(logits, dim=-1).clamp(min=1e-10)
                ent    = float(-torch.sum(probs * torch.log(probs)).item())
                return hs, ent
            except Exception as e:
                print(f"    extract failed: {e}", flush=True)
                return None, None

    def gen_text(question: str, context: str | None) -> str:
        if context:
            prompt = fmt_withcontext(question, context)
        else:
            prompt = fmt_nocontext(question)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=60, do_sample=False,
                               pad_token_id=tok.eos_token_id, use_cache=True)
        return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ── Collect bilateral oracle items ─────────────────────────────────────────
    print("  Collecting bilateral oracle …", flush=True)
    t0 = time.time()

    param_recs, ctx_recs = [], []

    for i, item in enumerate(pool):
        if len(param_recs) >= N_TARGET and len(ctx_recs) >= N_TARGET:
            break
        if i % 50 == 0:
            print(f"    [{i}] PARAM={len(param_recs)} CTX={len(ctx_recs)}"
                  f"  t={time.time()-t0:.0f}s", flush=True)

        # Nocontext pass
        nc_ans = gen_text(item["question"], None)
        nc_f1  = token_f1(nc_ans, item["answers"])

        hs_nc, ent_nc = extract_step1(item["question"], None)
        if hs_nc is None:
            continue

        # PARAM classification: nc_f1 >= 0.50
        if nc_f1 >= PARAM_MIN_F1 and len(param_recs) < N_TARGET:
            param_recs.append({
                "hs": hs_nc, "entropy": ent_nc,
                "nc_f1": nc_f1, "label": 1,
            })
            continue

        # CTX_DEP classification: nc_f1 <= 0.05 and ctx_f1 >= 0.50
        if nc_f1 <= CTX_MAX_NC and item.get("context") and len(ctx_recs) < N_TARGET:
            ctx_ans = gen_text(item["question"], item["context"])
            ctx_f1  = token_f1(ctx_ans, item["answers"])
            if ctx_f1 >= CTX_MIN_CTX:
                ctx_recs.append({
                    "hs": hs_nc, "entropy": ent_nc,
                    "nc_f1": nc_f1, "ctx_f1": ctx_f1, "label": 0,
                })

    print(f"  PARAM={len(param_recs)} CTX_DEP={len(ctx_recs)}", flush=True)

    if len(param_recs) < 20 or len(ctx_recs) < 20:
        print("  INSUFFICIENT SAMPLES — skipping", flush=True)
        del mdl; gc.collect(); torch.cuda.empty_cache()
        return None

    # ── Fisher+PCA64 AUROC ────────────────────────────────────────────────────
    recs = param_recs + ctx_recs
    ys   = np.array([r["label"] for r in recs])
    Xs   = np.array([r["hs"] for r in recs], dtype=np.float32)
    ents = np.array([r["entropy"] for r in recs])

    n_p  = len(param_recs)
    n_c  = len(ctx_recs)
    n_tr_p = int(n_p * TRAIN_FRAC)
    n_tr_c = int(n_c * TRAIN_FRAC)

    idx_tr = list(range(n_tr_p)) + list(range(n_p, n_p + n_tr_c))
    idx_te = list(range(n_tr_p, n_p)) + list(range(n_p + n_tr_c, n_p + n_c))

    X_tr, y_tr = Xs[idx_tr], ys[idx_tr]
    X_te, y_te = Xs[idx_te], ys[idx_te]
    e_tr, e_te = ents[idx_tr], ents[idx_te]

    fisher_auroc = None
    try:
        dim = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
        pca = PCA(n_components=dim, random_state=SEED)
        X_tr_p = pca.fit_transform(X_tr)
        X_te_p = pca.transform(X_te)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_tr_p, y_tr)
        sc_lda = lda.decision_function(X_te_p)
        fisher_auroc = round(float(roc_auc_score(y_te, sc_lda)), 4)
    except Exception as e:
        print(f"  Fisher LDA failed: {e}", flush=True)

    # Shuffled control
    fisher_shuf = None
    try:
        y_shuf = y_tr.copy(); np.random.shuffle(y_shuf)
        lda_sh = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda_sh.fit(X_tr_p, y_shuf)
        sc_sh = lda_sh.decision_function(X_te_p)
        fisher_shuf = round(float(roc_auc_score(y_te, sc_sh)), 4)
    except Exception:
        pass

    # Entropy AUROC (direct: label 1 = PARAM = higher param knowledge → lower NC entropy)
    entropy_auroc = None
    try:
        entropy_auroc = round(float(roc_auc_score(y_te, -e_te)), 4)
    except Exception as e:
        print(f"  Entropy AUROC failed: {e}", flush=True)

    print(f"  Fisher AUROC: {fisher_auroc}  (shuf: {fisher_shuf})", flush=True)
    print(f"  Entropy AUROC: {entropy_auroc}", flush=True)

    result = {
        "name":          cfg["name"],
        "model_id":      cfg["model_id"],
        "family":        cfg["family"],
        "n_params":      cfg["n_params"],
        "log10_params":  float(np.log10(cfg["n_params"])),
        "n_layers":      n_layers,
        "layer_idx":     layer_idx,
        "n_param_recs":  len(param_recs),
        "n_ctx_recs":    len(ctx_recs),
        "fisher_auroc":  fisher_auroc,
        "fisher_shuf":   fisher_shuf,
        "entropy_auroc": entropy_auroc,
    }

    del mdl; gc.collect(); torch.cuda.empty_cache()
    import time as _t; _t.sleep(2)   # let GPU settle
    return result


# ── Cross-model analysis ───────────────────────────────────────────────────────
def analyze(results: list) -> dict:
    print(f"\n=== Cross-Model Analysis ({len(results)} models) ===", flush=True)

    if len(results) < 3:
        return {"error": "fewer than 3 models completed", "n_models": len(results)}

    log_params    = np.array([r["log10_params"] for r in results])
    fisher_aurocs = np.array([r["fisher_auroc"] for r in results
                              if r.get("fisher_auroc") is not None])
    ent_aurocs    = np.array([r["entropy_auroc"] for r in results
                              if r.get("entropy_auroc") is not None])

    for r in results:
        print(f"  {r['name']:20s} | log_p={r['log10_params']:.2f} "
              f"| fisher={r.get('fisher_auroc','--')} "
              f"| entropy={r.get('entropy_auroc','--')}", flush=True)

    # Spearman correlations
    sp_fisher, _ = spearmanr(log_params[:len(fisher_aurocs)], fisher_aurocs)
    sp_entropy, _ = spearmanr(log_params[:len(ent_aurocs)], ent_aurocs)
    print(f"\n  Spearman ρ(Fisher, log_params):  {sp_fisher:.3f}", flush=True)
    print(f"  Spearman ρ(Entropy, log_params): {sp_entropy:.3f}", flush=True)

    # Family comparison at matched scale
    qwen_3b   = next((r for r in results if r["name"] == "qwen25_3b"), None)
    llama_3b  = next((r for r in results if r["name"] == "llama32_3b"), None)
    family_gap = None
    if qwen_3b and llama_3b and qwen_3b.get("fisher_auroc") and llama_3b.get("fisher_auroc"):
        family_gap = abs(qwen_3b["fisher_auroc"] - llama_3b["fisher_auroc"])
        print(f"  Qwen-3B vs Llama-3B Fisher gap: {family_gap:.3f}", flush=True)

    # Is Fisher AUROC monotone?
    pairs = [(r["log10_params"], r["fisher_auroc"]) for r in results
             if r.get("fisher_auroc") is not None]
    pairs.sort(key=lambda x: x[0])
    monotone_fisher = all(pairs[i][1] <= pairs[i+1][1] for i in range(len(pairs)-1))

    # Verdict
    def verdict():
        if sp_fisher >= 0.80:
            return "MONOTONE_FISHER"
        if sp_entropy >= 0.80:
            return "MONOTONE_ENTROPY"
        if sp_fisher <= 0.40 and sp_entropy <= 0.40:
            return "FLAT_BOTH"
        if family_gap is not None and family_gap >= 0.10:
            return "FAMILY_DIVERGENCE"
        if len(results) < 4:
            return "INCONCLUSIVE"
        return "AMBIGUOUS"

    verd = verdict()
    print(f"\n  VERDICT: {verd}", flush=True)

    return {
        "n_models":          len(results),
        "spearman_fisher":   round(float(sp_fisher), 4),
        "spearman_entropy":  round(float(sp_entropy), 4),
        "family_gap_3b":     round(family_gap, 4) if family_gap is not None else None,
        "monotone_fisher":   bool(monotone_fisher),
        "verdict":           verd,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"EXP-E: Observability Scaling v1", flush=True)
    print(f"{'='*60}", flush=True)

    pool = load_pool(POOL_SIZE)

    per_model_results = []
    for cfg in MODELS:
        res = run_model(cfg, pool)
        if res is not None:
            per_model_results.append(res)
            # Intermediate save
            with open("scale_obs_v1_results.json", "w") as f:
                json.dump({
                    "experiment": "EXP_E_SCALE_OBS_V1",
                    "config": {
                        "pool_size": POOL_SIZE,
                        "n_target": N_TARGET,
                        "train_frac": TRAIN_FRAC,
                        "pca_dim": PCA_DIM,
                        "seed": SEED,
                    },
                    "models": per_model_results,
                    "analysis": analyze(per_model_results),
                }, f, indent=2)
            print(f"  Intermediate save after {cfg['name']}", flush=True)

    if len(per_model_results) == 0:
        print("FATAL: no models completed", flush=True)
        sys.exit(1)

    analysis = analyze(per_model_results)

    final = {
        "experiment": "EXP_E_SCALE_OBS_V1",
        "config": {
            "pool_size": POOL_SIZE,
            "n_target":  N_TARGET,
            "train_frac": TRAIN_FRAC,
            "pca_dim":   PCA_DIM,
            "seed":      SEED,
        },
        "models":   per_model_results,
        "analysis": analysis,
    }

    with open("scale_obs_v1_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFinal results saved to scale_obs_v1_results.json", flush=True)


if __name__ == "__main__":
    main()

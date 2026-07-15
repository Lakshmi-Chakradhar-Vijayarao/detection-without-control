#!/usr/bin/env python3
"""
scale_extension_v1.py — Scale Extension: Does bilateral oracle AUROC survive
                         above the 3B Goldilocks ceiling?

SCIENTIFIC QUESTION:
  EXP-E (scale_obs_v1) established a Goldilocks zone: Qwen2.5-0.5B fails
  (too few PARAM items), Qwen2.5-3B fails (CTX_DEP yield near zero — the
  model knows too much parametrically).  The sweet spot is ~1B-2B params.

  Does the bilateral oracle Fisher+PCA64 probe still work above this ceiling?
  Testing Qwen2.5-7B-Instruct: a larger model that knows more parametrically,
  so CTX_DEP yield on standard TriviaQA will be low.

DESIGN:
  Model: Qwen/Qwen2.5-7B-Instruct (~14 GB float16, tight on T4 16 GB)
  Dataset: TriviaQA rc.wikipedia validation, pool of 5000 items
  Protocol:
    PARAM   : nocontext F1 ≥ 0.50 OR answer_contains match
    CTX_DEP : nocontext F1 ≤ 0.05 AND answer_contains=False
              AND withcontext F1 ≥ 0.50 OR answer_contains=True
  Targets:  N_PARAM=50, N_CTX_DEP=50 (minimum 20/class to report)
  Probe:    Fisher+PCA64 at layer_idx = int(n_layers * 26/28)
            (same fractional depth as L26 in 1.5B; ~93% depth)

DECISION GATE:
  AUROC_SURVIVED    : fisher_auroc ≥ 0.65 (comparable to 1.5B result 0.73)
  AUROC_DEGRADED    : 0.55 ≤ fisher_auroc < 0.65 (partial signal above ceiling)
  AUROC_LOST        : fisher_auroc < 0.55 (probe fails above Goldilocks zone)
  INSUFFICIENT_DATA : fewer than 20/class collected from 5000-item pool

MEMORY MANAGEMENT:
  Loaded in 4-bit quantization (bitsandbytes NF4) — reduces 14 GB float16
  to ~4 GB, well within T4 16 GB. Hidden states are computed in float16
  internally; Fisher probe operates on those states, not weights, so
  quantization does not affect probe validity.
  KV cache enabled.

GPU: T4 (16 GB). Expected ~4-5h.
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID          = "Qwen/Qwen2.5-7B-Instruct"
POOL_SIZE         = 5_000
N_PARAM_TARGET    = 50
N_CTX_DEP_TARGET  = 50
MIN_CLASS_N       = 20        # minimum per class to report AUROC
PARAM_MIN_F1      = 0.50
CTX_MAX_NC        = 0.05
CTX_MIN_CTX       = 0.50
TRAIN_FRAC        = 0.75
PCA_DIM           = 64
SAVE_EVERY        = 10        # intermediate save every N collected items (per class)
SEED              = 42

RESULTS_FILE      = "/kaggle/working/scale_ext_results.json"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── GPU fast-fail ──────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("GPU required — no CUDA device found")
DEVICE = "cuda"
_gpu_name = torch.cuda.get_device_name(0)
_sm_major, _sm_minor = torch.cuda.get_device_capability(0)
_sm = _sm_major * 10 + _sm_minor
print(f"GPU: {_gpu_name}  (sm_{_sm})", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
# T4 = sm_75; P100 = sm_60. Fail fast on unsupported hardware.
assert _sm >= 70, (
    f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. "
    "Re-run to get a different GPU."
)


# ── HF token ──────────────────────────────────────────────────────────────────
def _get_hf_token() -> str | None:
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
def load_pool(n: int) -> list:
    print(f"Loading TriviaQA pool (n={n}) …", flush=True)
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      trust_remote_code=True)
    items = []
    for ex in ds:
        ep  = ex.get("entity_pages", {})
        ctx = (ep.get("wiki_context") or [""])[0][:800] if ep else ""
        ans = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        items.append({"question": ex["question"], "context": ctx, "answers": ans})
        if len(items) >= n:
            break
    random.shuffle(items)
    print(f"  Pool ready: {len(items)} items", flush=True)
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


def answer_contains(pred: str, golds: list) -> bool:
    return any(g.lower() in pred.lower() for g in golds)


def fmt_nocontext(q: str) -> str:
    return f"Answer in one short phrase.\nQuestion: {q}\nAnswer:"


def fmt_withcontext(q: str, ctx: str) -> str:
    return (f"Use the passage to answer.\n"
            f"Passage: {ctx}\nQuestion: {q}\nAnswer:")


# ── Score helper ───────────────────────────────────────────────────────────────
def score(pred: str, golds: list) -> float:
    """Return max(token_f1, answer_contains) as a float."""
    f1 = token_f1(pred, golds)
    ac = 1.0 if answer_contains(pred, golds) else 0.0
    return max(f1, ac)


# ── Intermediate save ──────────────────────────────────────────────────────────
def save_intermediate(state: dict) -> None:
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"  [warn] intermediate save failed: {e}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"Scale Extension v1 — Qwen2.5-7B-Instruct", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    pool = load_pool(POOL_SIZE)

    # ── Load model ─────────────────────────────────────────────────────────────
    tok_hf = _get_hf_token()
    if tok_hf:
        from huggingface_hub import login
        login(token=tok_hf, add_to_git_credential=False)

    print(f"\nLoading {MODEL_ID} …", flush=True)
    t_load = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit quantization: 7B float16 = 14 GB > T4 budget; NF4 reduces to ~4 GB
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",        # required for bitsandbytes 4-bit
        trust_remote_code=True,
    ).eval()

    n_layers  = mdl.config.num_hidden_layers
    # Use same fractional depth as L26 in 1.5B (26/28 ≈ 93% depth)
    layer_idx = int(n_layers * 26 / 28)
    print(f"  n_layers={n_layers}, layer_idx={layer_idx} "
          f"({layer_idx/n_layers*100:.0f}% depth)", flush=True)
    print(f"  Load time: {time.time()-t_load:.1f}s", flush=True)
    print(f"  VRAM after load: "
          f"{torch.cuda.memory_allocated()/1e9:.1f} GB allocated, "
          f"{torch.cuda.memory_reserved()/1e9:.1f} GB reserved", flush=True)

    # ── Inference helpers ──────────────────────────────────────────────────────
    def gen_text(prompt: str) -> str:
        inp = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=512).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(
                **inp,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        return tok.decode(
            out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def extract_step1(prompt: str):
        """Single max_new_tokens=1 forward pass; return (hidden_state, entropy)."""
        inp = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=512).to(DEVICE)
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
                hs     = out.hidden_states[0][layer_idx][0, -1, :].float().cpu().numpy()
                logits = out.scores[0][0].float()
                probs  = torch.softmax(logits, dim=-1).clamp(min=1e-10)
                ent    = float(-torch.sum(probs * torch.log(probs)).item())
                return hs, ent
            except Exception as e:
                print(f"    extract_step1 failed: {e}", flush=True)
                return None, None

    # ── Bilateral oracle collection ────────────────────────────────────────────
    print("\nCollecting bilateral oracle items …", flush=True)
    t0 = time.time()

    param_recs: list  = []
    ctx_recs:   list  = []
    n_scanned         = 0
    last_save_p       = 0
    last_save_c       = 0

    state_base = {
        "experiment":      "SCALE_EXTENSION_V1",
        "model_id":        MODEL_ID,
        "n_layers":        n_layers,
        "layer_idx":       layer_idx,
        "pool_size":       POOL_SIZE,
        "n_param_target":  N_PARAM_TARGET,
        "n_ctx_dep_target": N_CTX_DEP_TARGET,
        "pca_dim":         PCA_DIM,
        "seed":            SEED,
        "status":          "in_progress",
    }

    for item in pool:
        if len(param_recs) >= N_PARAM_TARGET and len(ctx_recs) >= N_CTX_DEP_TARGET:
            break

        n_scanned += 1
        if n_scanned % 50 == 0:
            print(f"  [{n_scanned}] PARAM={len(param_recs)} "
                  f"CTX_DEP={len(ctx_recs)}  t={time.time()-t0:.0f}s",
                  flush=True)
            torch.cuda.empty_cache()

        q   = item["question"]
        ctx = item["context"]
        ans = item["answers"]

        # ── Nocontext pass ─────────────────────────────────────────────────────
        nc_text = gen_text(fmt_nocontext(q))
        nc_sc   = score(nc_text, ans)

        hs_nc, ent_nc = extract_step1(fmt_nocontext(q))
        if hs_nc is None:
            continue

        # ── PARAM classification ───────────────────────────────────────────────
        if nc_sc >= PARAM_MIN_F1 and len(param_recs) < N_PARAM_TARGET:
            param_recs.append({
                "hs":      hs_nc.tolist(),
                "entropy": ent_nc,
                "nc_score": nc_sc,
                "label":   1,
            })
            # Intermediate save every SAVE_EVERY PARAM items
            if len(param_recs) - last_save_p >= SAVE_EVERY:
                save_intermediate({
                    **state_base,
                    "n_param_collected":   len(param_recs),
                    "n_ctx_dep_collected": len(ctx_recs),
                    "n_scanned":           n_scanned,
                })
                last_save_p = len(param_recs)
            continue

        # ── CTX_DEP classification ─────────────────────────────────────────────
        if nc_sc <= CTX_MAX_NC and ctx and len(ctx_recs) < N_CTX_DEP_TARGET:
            ctx_text = gen_text(fmt_withcontext(q, ctx))
            ctx_sc   = score(ctx_text, ans)
            if ctx_sc >= CTX_MIN_CTX:
                ctx_recs.append({
                    "hs":       hs_nc.tolist(),
                    "entropy":  ent_nc,
                    "nc_score": nc_sc,
                    "ctx_score": ctx_sc,
                    "label":    0,
                })
                # Intermediate save every SAVE_EVERY CTX_DEP items
                if len(ctx_recs) - last_save_c >= SAVE_EVERY:
                    save_intermediate({
                        **state_base,
                        "n_param_collected":   len(param_recs),
                        "n_ctx_dep_collected": len(ctx_recs),
                        "n_scanned":           n_scanned,
                    })
                    last_save_c = len(ctx_recs)

    print(f"\nCollection complete: PARAM={len(param_recs)} "
          f"CTX_DEP={len(ctx_recs)}  scanned={n_scanned}", flush=True)

    # ── Goldilocks ceiling check ───────────────────────────────────────────────
    if len(param_recs) < MIN_CLASS_N or len(ctx_recs) < MIN_CLASS_N:
        verdict = "INSUFFICIENT_DATA"
        print(f"  VERDICT: {verdict} — fewer than {MIN_CLASS_N}/class", flush=True)
        result = {
            **state_base,
            "status":            "complete",
            "n_param_collected": len(param_recs),
            "n_ctx_dep_collected": len(ctx_recs),
            "n_scanned":         n_scanned,
            "verdict":           verdict,
            "fisher_auroc":      None,
            "entropy_auroc":     None,
            "fisher_shuf":       None,
            "note": (
                f"CTX_DEP yield too low at 7B scale ({len(ctx_recs)} items). "
                "Confirms Goldilocks ceiling — bilateral oracle inapplicable "
                "above ~3B on standard TriviaQA without domain-specific "
                "hard questions."
            ),
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {RESULTS_FILE}", flush=True)
        del mdl; gc.collect(); torch.cuda.empty_cache()
        return

    # ── Fisher+PCA64 probe ────────────────────────────────────────────────────
    print("\nFitting Fisher+PCA64 probe …", flush=True)

    recs  = param_recs + ctx_recs
    ys    = np.array([r["label"]   for r in recs])
    Xs    = np.array([r["hs"]      for r in recs], dtype=np.float32)
    ents  = np.array([r["entropy"] for r in recs])

    n_p   = len(param_recs)
    n_c   = len(ctx_recs)
    n_tr_p = int(n_p * TRAIN_FRAC)
    n_tr_c = int(n_c * TRAIN_FRAC)

    idx_tr = list(range(n_tr_p)) + list(range(n_p, n_p + n_tr_c))
    idx_te = list(range(n_tr_p, n_p)) + list(range(n_p + n_tr_c, n_p + n_c))

    X_tr, y_tr = Xs[idx_tr], ys[idx_tr]
    X_te, y_te = Xs[idx_te], ys[idx_te]
    e_te       = ents[idx_te]

    fisher_auroc = None
    fisher_shuf  = None
    X_tr_pca     = None

    try:
        dim   = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
        pca   = PCA(n_components=dim, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_tr_pca, y_tr)
        sc  = lda.decision_function(X_te_pca)
        fisher_auroc = round(float(roc_auc_score(y_te, sc)), 4)
        print(f"  Fisher AUROC: {fisher_auroc}", flush=True)
    except Exception as e:
        print(f"  Fisher LDA failed: {e}", flush=True)

    # Shuffled-label control
    if X_tr_pca is not None:
        try:
            y_shuf = y_tr.copy()
            np.random.shuffle(y_shuf)
            lda_sh = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda_sh.fit(X_tr_pca, y_shuf)
            sc_sh  = lda_sh.decision_function(X_te_pca)
            fisher_shuf = round(float(roc_auc_score(y_te, sc_sh)), 4)
            print(f"  Fisher AUROC (shuffled): {fisher_shuf}", flush=True)
        except Exception as e:
            print(f"  Shuffled control failed: {e}", flush=True)

    # Entropy AUROC
    entropy_auroc = None
    try:
        # PARAM (label=1) has lower entropy (model is confident) → negate
        entropy_auroc = round(float(roc_auc_score(y_te, -e_te)), 4)
        print(f"  Entropy AUROC: {entropy_auroc}", flush=True)
    except Exception as e:
        print(f"  Entropy AUROC failed: {e}", flush=True)

    # ── Verdict ────────────────────────────────────────────────────────────────
    if fisher_auroc is None:
        verdict = "PROBE_FAILED"
    elif fisher_auroc >= 0.65:
        verdict = "AUROC_SURVIVED"
    elif fisher_auroc >= 0.55:
        verdict = "AUROC_DEGRADED"
    else:
        verdict = "AUROC_LOST"

    print(f"\n  VERDICT: {verdict}", flush=True)

    # ── Reference comparison ───────────────────────────────────────────────────
    # Known 1.5B-Instruct bilateral oracle AUROC from the core program
    reference_1p5b = {"model": "Qwen2.5-1.5B-Instruct", "fisher_auroc": 0.7300}
    print(f"  Reference (1.5B-Instruct): {reference_1p5b['fisher_auroc']}", flush=True)
    if fisher_auroc is not None:
        delta = round(fisher_auroc - reference_1p5b["fisher_auroc"], 4)
        print(f"  Delta vs 1.5B: {delta:+.4f}", flush=True)
    else:
        delta = None

    # ── Final output ───────────────────────────────────────────────────────────
    result = {
        **state_base,
        "status":                "complete",
        "n_param_collected":     len(param_recs),
        "n_ctx_dep_collected":   len(ctx_recs),
        "n_scanned":             n_scanned,
        "train_frac":            TRAIN_FRAC,
        "n_train_param":         n_tr_p,
        "n_train_ctx":           n_tr_c,
        "n_test_param":          n_p - n_tr_p,
        "n_test_ctx":            n_c - n_tr_c,
        "fisher_auroc":          fisher_auroc,
        "fisher_shuf":           fisher_shuf,
        "entropy_auroc":         entropy_auroc,
        "verdict":               verdict,
        "reference_1p5b":        reference_1p5b,
        "delta_vs_1p5b":         delta,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)

    del mdl; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

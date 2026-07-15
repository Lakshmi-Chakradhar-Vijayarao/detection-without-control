#!/usr/bin/env python3
"""
exp_l_stage_sweep_v1.py — EXP-L: Training Stage Sweep
                           Base → RLHF/Instruct → Reasoning

SCIENTIFIC QUESTION:
  How does the bilateral oracle Fisher+PCA64 AUROC change across LM training
  stages?  Specifically:
    Stage 1 (Base):      Qwen2.5-1.5B   (base, no instruction tuning)
    Stage 2 (Instruct):  Qwen2.5-1.5B-Instruct  (RLHF / supervised fine-tuning)
    Stage 3 (Reasoning): DeepSeek-R1-Distill-Qwen-1.5B  (GRPO reasoning fine-tune)

  Stages 2 and 3 are already measured in this program; their AUROCs are
  hardcoded as constants.  Stage 1 (base) runs LIVE on Kaggle.

C026 EXTENSION:
  C026 (from pythia_sweep) states: pure autoregressive base LMs that don't
  follow QA instructions produce CTX_DEP=0 because they never answer with
  context even when context is provided.  Pythia was the example.

  Qwen2.5-1.5B base is a different situation: it was trained on a large,
  instruction-format-rich corpus and may partially follow QA prompts.  This
  experiment tests whether the bilateral oracle is applicable to a capable
  base model.

  Inapplicability criterion: if CTX_DEP < 5 after scanning 2000 items,
  report BILATERAL_ORACLE_INAPPLICABLE and document the C026 extension.

DESIGN — STAGE 1 (live):
  Model:   Qwen/Qwen2.5-1.5B  (base, not instruct)
  Prompt format (no chat template):
    nocontext:   "Question: {q}\\nAnswer:"
    withcontext: "Context: {ctx}\\nQuestion: {q}\\nAnswer:"
  Layer:   26 (same as throughout the program, 28-layer arch)
  Pool:    3000 items from TriviaQA rc.wikipedia validation
  Targets: N_PARAM=50, N_CTX_DEP=50
           Inapplicability check at 2000 items if CTX_DEP < 5

DESIGN — STAGES 2 & 3 (hardcoded from prior results):
  Stage 2: Fisher AUROC = 0.7300
    Source: false_certainty_v2 bilateral oracle calibration;
            consistent across scale_obs_v1, large_n_validation
  Stage 3: Fisher AUROC = 0.7600
    Source: reasoning_geometry_b_results (Cal AUROC at L26, EXP-B)

OUTPUT:
  JSON with per-stage results and comparison table.
  Saved to /kaggle/working/exp_l_stage_sweep_results.json

GPU: T4. Expected ~2-3h (Stage 1 only).
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
STAGE1_MODEL_ID   = "Qwen/Qwen2.5-1.5B"    # base, no instruct
LAYER_IDX         = 26                      # consistent with full program
POOL_SIZE         = 3_000
N_PARAM_TARGET    = 50
N_CTX_DEP_TARGET  = 50
MIN_CLASS_N       = 20
INAPP_CHECK_N     = 2_000                   # check inapplicability after N items
INAPP_CTX_THRESH  = 5                       # if CTX_DEP < this → INAPPLICABLE
PARAM_MIN_F1      = 0.50
CTX_MAX_NC        = 0.05
CTX_MIN_CTX       = 0.50
TRAIN_FRAC        = 0.75
PCA_DIM           = 64
SAVE_EVERY        = 10
SEED              = 42

RESULTS_FILE      = "/kaggle/working/exp_l_stage_sweep_results.json"

# ── Hardcoded prior results ────────────────────────────────────────────────────
# These stages were measured in earlier program runs; they are NOT rerun here.
STAGE2_KNOWN = {
    "stage":         2,
    "label":         "Instruct (RLHF)",
    "model_id":      "Qwen/Qwen2.5-1.5B-Instruct",
    "fisher_auroc":  0.7300,
    "source":        "false_certainty_v2 bilateral oracle calibration; "
                     "replicated in scale_obs_v1 and large_n_validation",
    "live":          False,
}

STAGE3_KNOWN = {
    "stage":         3,
    "label":         "Reasoning (GRPO distill)",
    "model_id":      "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "fisher_auroc":  0.7600,
    "source":        "reasoning_geometry_v1 (EXP-B Cal AUROC at L26)",
    "live":          False,
}

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
    print(f"Loading TriviaQA rc.wikipedia pool (n={n}) …", flush=True)
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
    pt   = set(pred.lower().split())
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


def score(pred: str, golds: list) -> float:
    """Combined token_f1 / answer_contains score."""
    return max(token_f1(pred, golds), 1.0 if answer_contains(pred, golds) else 0.0)


# ── Base-model prompt format (no chat template) ────────────────────────────────
def fmt_nocontext_base(q: str) -> str:
    return f"Question: {q}\nAnswer:"


def fmt_withcontext_base(q: str, ctx: str) -> str:
    return f"Context: {ctx}\nQuestion: {q}\nAnswer:"


# ── Intermediate save ──────────────────────────────────────────────────────────
def save_intermediate(state: dict) -> None:
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"  [warn] save failed: {e}", flush=True)


# ── Stage 1: live run on Qwen2.5-1.5B base ────────────────────────────────────
def run_stage1(pool: list) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 1 — {STAGE1_MODEL_ID}  (base, no instruct)", flush=True)
    print(f"{'='*60}", flush=True)

    tok_hf = _get_hf_token()
    if tok_hf:
        from huggingface_hub import login
        login(token=tok_hf, add_to_git_credential=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(STAGE1_MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        STAGE1_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    n_layers = mdl.config.num_hidden_layers
    print(f"  n_layers={n_layers}, layer_idx={LAYER_IDX}", flush=True)
    print(f"  Load time: {time.time()-t_load:.1f}s", flush=True)

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
        # Decode only newly generated tokens
        gen_ids = out[0][inp["input_ids"].shape[1]:]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()

    def extract_step1(prompt: str):
        """Single max_new_tokens=1 pass; return (hidden_state_at_L26, entropy)."""
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
                hs     = out.hidden_states[0][LAYER_IDX][0, -1, :].float().cpu().numpy()
                logits = out.scores[0][0].float()
                probs  = torch.softmax(logits, dim=-1).clamp(min=1e-10)
                ent    = float(-torch.sum(probs * torch.log(probs)).item())
                return hs, ent
            except Exception as e:
                print(f"    extract_step1 failed: {e}", flush=True)
                return None, None

    # ── Bilateral oracle collection ────────────────────────────────────────────
    print("  Collecting bilateral oracle items …", flush=True)
    t0 = time.time()

    param_recs: list = []
    ctx_recs:   list = []
    n_scanned        = 0
    last_save_p      = 0
    last_save_c      = 0
    inapp_checked    = False

    state_base = {
        "experiment":    "EXP_L_STAGE_SWEEP_V1",
        "stage":         1,
        "label":         "Base (pre-RLHF)",
        "model_id":      STAGE1_MODEL_ID,
        "n_layers":      n_layers,
        "layer_idx":     LAYER_IDX,
        "pool_size":     POOL_SIZE,
        "n_param_target": N_PARAM_TARGET,
        "n_ctx_dep_target": N_CTX_DEP_TARGET,
        "pca_dim":       PCA_DIM,
        "seed":          SEED,
        "live":          True,
        "status":        "in_progress",
    }

    for item in pool:
        if len(param_recs) >= N_PARAM_TARGET and len(ctx_recs) >= N_CTX_DEP_TARGET:
            break

        n_scanned += 1

        # C026 extension check: inapplicability after INAPP_CHECK_N items
        if (not inapp_checked
                and n_scanned >= INAPP_CHECK_N
                and len(ctx_recs) < INAPP_CTX_THRESH):
            inapp_checked = True
            print(f"\n  C026-EXTENSION CHECK at n_scanned={n_scanned}: "
                  f"CTX_DEP={len(ctx_recs)} < {INAPP_CTX_THRESH}", flush=True)
            print("  → BILATERAL_ORACLE_INAPPLICABLE for base model", flush=True)
            print("  → Qwen2.5-1.5B base does not reliably follow context-grounded QA.",
                  flush=True)
            del mdl; gc.collect(); torch.cuda.empty_cache()
            return {
                **state_base,
                "status":              "complete",
                "n_param_collected":   len(param_recs),
                "n_ctx_dep_collected": len(ctx_recs),
                "n_scanned":           n_scanned,
                "verdict":             "BILATERAL_ORACLE_INAPPLICABLE",
                "fisher_auroc":        None,
                "entropy_auroc":       None,
                "fisher_shuf":         None,
                "c026_note": (
                    f"C026 EXTENDED: After scanning {n_scanned} items, "
                    f"only {len(ctx_recs)} CTX_DEP items found "
                    f"(threshold={INAPP_CTX_THRESH}). "
                    "Qwen2.5-1.5B (base) does not reliably use provided "
                    "context to answer questions it failed without context. "
                    "The bilateral oracle protocol requires a model that "
                    "follows QA instructions; base models without instruction "
                    "tuning may fail to condition on context even when present."
                ),
            }

        if n_scanned % 100 == 0:
            print(f"  [{n_scanned}] PARAM={len(param_recs)} "
                  f"CTX_DEP={len(ctx_recs)}  t={time.time()-t0:.0f}s",
                  flush=True)
            torch.cuda.empty_cache()

        q   = item["question"]
        ctx = item["context"]
        ans = item["answers"]

        # Base model uses plain prompt, no chat template
        nc_text = gen_text(fmt_nocontext_base(q))
        nc_sc   = score(nc_text, ans)

        hs_nc, ent_nc = extract_step1(fmt_nocontext_base(q))
        if hs_nc is None:
            continue

        # PARAM: model answers correctly without context
        if nc_sc >= PARAM_MIN_F1 and len(param_recs) < N_PARAM_TARGET:
            param_recs.append({
                "hs":       hs_nc.tolist(),
                "entropy":  ent_nc,
                "nc_score": nc_sc,
                "label":    1,
            })
            if len(param_recs) - last_save_p >= SAVE_EVERY:
                save_intermediate({
                    **state_base,
                    "n_param_collected":   len(param_recs),
                    "n_ctx_dep_collected": len(ctx_recs),
                    "n_scanned":           n_scanned,
                })
                last_save_p = len(param_recs)
            continue

        # CTX_DEP: fails without context, succeeds with context
        if nc_sc <= CTX_MAX_NC and ctx and len(ctx_recs) < N_CTX_DEP_TARGET:
            ctx_text = gen_text(fmt_withcontext_base(q, ctx))
            ctx_sc   = score(ctx_text, ans)
            if ctx_sc >= CTX_MIN_CTX:
                ctx_recs.append({
                    "hs":        hs_nc.tolist(),
                    "entropy":   ent_nc,
                    "nc_score":  nc_sc,
                    "ctx_score": ctx_sc,
                    "label":     0,
                })
                if len(ctx_recs) - last_save_c >= SAVE_EVERY:
                    save_intermediate({
                        **state_base,
                        "n_param_collected":   len(param_recs),
                        "n_ctx_dep_collected": len(ctx_recs),
                        "n_scanned":           n_scanned,
                    })
                    last_save_c = len(ctx_recs)

    print(f"\n  Collection done: PARAM={len(param_recs)} "
          f"CTX_DEP={len(ctx_recs)}  scanned={n_scanned}", flush=True)

    # Insufficient data check
    if len(param_recs) < MIN_CLASS_N or len(ctx_recs) < MIN_CLASS_N:
        verdict = (
            "BILATERAL_ORACLE_INAPPLICABLE"
            if len(ctx_recs) < MIN_CLASS_N
            else "INSUFFICIENT_DATA"
        )
        note = (
            f"C026 EXTENDED: Only {len(ctx_recs)} CTX_DEP items after full pool. "
            "Base model does not reliably follow context instructions."
            if len(ctx_recs) < MIN_CLASS_N
            else f"Only {len(param_recs)} PARAM items found (need {MIN_CLASS_N})."
        )
        del mdl; gc.collect(); torch.cuda.empty_cache()
        return {
            **state_base,
            "status":              "complete",
            "n_param_collected":   len(param_recs),
            "n_ctx_dep_collected": len(ctx_recs),
            "n_scanned":           n_scanned,
            "verdict":             verdict,
            "fisher_auroc":        None,
            "entropy_auroc":       None,
            "fisher_shuf":         None,
            "c026_note":           note,
        }

    # ── Fisher+PCA64 probe ────────────────────────────────────────────────────
    print("  Fitting Fisher+PCA64 probe …", flush=True)

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

    entropy_auroc = None
    try:
        entropy_auroc = round(float(roc_auc_score(y_te, -e_te)), 4)
        print(f"  Entropy AUROC: {entropy_auroc}", flush=True)
    except Exception as e:
        print(f"  Entropy AUROC failed: {e}", flush=True)

    # Verdict
    if fisher_auroc is None:
        verdict = "PROBE_FAILED"
    elif fisher_auroc >= 0.65:
        verdict = "ORACLE_APPLICABLE"
    elif fisher_auroc >= 0.55:
        verdict = "WEAK_SIGNAL"
    else:
        verdict = "BILATERAL_ORACLE_MARGINAL"

    print(f"  VERDICT: {verdict}", flush=True)

    del mdl; gc.collect(); torch.cuda.empty_cache()

    return {
        **state_base,
        "status":              "complete",
        "n_param_collected":   n_p,
        "n_ctx_dep_collected": n_c,
        "n_scanned":           n_scanned,
        "train_frac":          TRAIN_FRAC,
        "n_train_param":       n_tr_p,
        "n_train_ctx":         n_tr_c,
        "n_test_param":        n_p - n_tr_p,
        "n_test_ctx":          n_c - n_tr_c,
        "fisher_auroc":        fisher_auroc,
        "fisher_shuf":         fisher_shuf,
        "entropy_auroc":       entropy_auroc,
        "verdict":             verdict,
    }


# ── Stage comparison analysis ─────────────────────────────────────────────────
def analyze_stages(stage1: dict, stage2: dict, stage3: dict) -> dict:
    print(f"\n{'='*60}", flush=True)
    print("STAGE COMPARISON TABLE", flush=True)
    print(f"{'='*60}", flush=True)

    header = f"  {'Stage':<6} {'Label':<28} {'Fisher AUROC':>12}  {'Note'}"
    print(header, flush=True)
    print(f"  {'-'*70}", flush=True)

    def auroc_str(d):
        a = d.get("fisher_auroc")
        if a is None:
            v = d.get("verdict", "N/A")
            return f"{'N/A':>12}  [{v}]"
        return f"{a:>12.4f}"

    for s in [stage1, stage2, stage3]:
        live_tag = "(live)" if s.get("live", True) else "(prior)"
        print(f"  {s['stage']:<6} {s['label']:<28} {auroc_str(s)}  {live_tag}",
              flush=True)

    # Quantify training-stage effect
    s1_a = stage1.get("fisher_auroc")
    s2_a = stage2.get("fisher_auroc")
    s3_a = stage3.get("fisher_auroc")

    trend = None
    if s1_a is not None and s2_a is not None:
        base_to_instruct = round(s2_a - s1_a, 4)
        print(f"\n  Base → Instruct delta: {base_to_instruct:+.4f}", flush=True)
    else:
        base_to_instruct = None
        print("\n  Base → Instruct delta: N/A (Stage 1 inapplicable)", flush=True)

    if s2_a is not None and s3_a is not None:
        instruct_to_reason = round(s3_a - s2_a, 4)
        print(f"  Instruct → Reasoning delta: {instruct_to_reason:+.4f}", flush=True)
    else:
        instruct_to_reason = None

    # Overall trend across available stages
    available = [(s["stage"], s["fisher_auroc"])
                 for s in [stage1, stage2, stage3]
                 if s.get("fisher_auroc") is not None]
    if len(available) >= 2:
        auroc_vals = [a for _, a in available]
        if auroc_vals == sorted(auroc_vals):
            trend = "MONOTONE_INCREASE_WITH_TRAINING"
        elif auroc_vals == sorted(auroc_vals, reverse=True):
            trend = "MONOTONE_DECREASE_WITH_TRAINING"
        else:
            trend = "NON_MONOTONE"
    elif len(available) == 1:
        trend = "SINGLE_STAGE_ONLY"
    else:
        trend = "NO_DATA"

    print(f"\n  Trend: {trend}", flush=True)

    return {
        "stages_completed":        len(available),
        "base_to_instruct_delta":  base_to_instruct,
        "instruct_to_reason_delta": instruct_to_reason,
        "trend":                   trend,
        "stage1_applicable": stage1.get("verdict") not in (
            "BILATERAL_ORACLE_INAPPLICABLE", "PROBE_FAILED",
            "INSUFFICIENT_DATA", None
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print("EXP-L: Training Stage Sweep v1", flush=True)
    print("Base → Instruct → Reasoning", flush=True)
    print(f"{'='*60}", flush=True)

    # Stage 2 & 3 from prior program results (hardcoded)
    stage2 = STAGE2_KNOWN.copy()
    stage3 = STAGE3_KNOWN.copy()

    print(f"\nStage 2 ({stage2['model_id']}): "
          f"fisher_auroc={stage2['fisher_auroc']} [prior result, not rerun]",
          flush=True)
    print(f"Stage 3 ({stage3['model_id']}): "
          f"fisher_auroc={stage3['fisher_auroc']} [prior result, not rerun]",
          flush=True)

    # Load data once for Stage 1
    pool = load_pool(POOL_SIZE)

    # Stage 1 — live run
    stage1 = run_stage1(pool)

    # Analysis
    analysis = analyze_stages(stage1, stage2, stage3)

    # Final output
    result = {
        "experiment": "EXP_L_STAGE_SWEEP_V1",
        "description": "Bilateral oracle Fisher AUROC across training stages: "
                       "Base → Instruct → Reasoning",
        "config": {
            "stage1_model":    STAGE1_MODEL_ID,
            "layer_idx":       LAYER_IDX,
            "pool_size":       POOL_SIZE,
            "n_param_target":  N_PARAM_TARGET,
            "n_ctx_dep_target": N_CTX_DEP_TARGET,
            "pca_dim":         PCA_DIM,
            "seed":            SEED,
            "inapp_check_n":   INAPP_CHECK_N,
            "inapp_ctx_thresh": INAPP_CTX_THRESH,
        },
        "stages": {
            "stage1_base":      stage1,
            "stage2_instruct":  stage2,
            "stage3_reasoning": stage3,
        },
        "analysis": analysis,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()

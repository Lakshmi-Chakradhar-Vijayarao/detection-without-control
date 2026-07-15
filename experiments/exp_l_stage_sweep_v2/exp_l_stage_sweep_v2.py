#!/usr/bin/env python3
"""
exp_l_stage_sweep_v2.py — EXP_L_STAGE_SWEEP_V2
Clean training-stage comparison at matched N=200/class.

SCIENTIFIC QUESTION (Layer 2 — Laws):
  How does bilateral oracle legibility change across training stages?
  Does the Fisher gap (L2) emerge from instruction tuning, RLHF, or is it present in base models?
  What does this imply about which training objective creates the epistemic geometry?

WHY V2:
  EXP_L (v1) was confounded by N differences:
    Stage 1 (base):       N=50/class
    Stage 2 (instruct):   N=197/class
    Stage 3 (reasoning):  N=200/class
  Those N differences make the AUROC comparison invalid.
  V2 uses matched N=200/class across all stages.

DESIGN:
  Three models from a single architecture family (Qwen) at three training stages:
    Stage 1 BASE:      Qwen/Qwen2.5-1.5B         (pure pretrain, autoregressive)
    Stage 2 INSTRUCT:  Qwen/Qwen2.5-1.5B-Instruct (SFT + RLHF)
    Stage 3 REASONING: Qwen/QwQ-32B is too big — use Qwen/Qwen3-0.6B or DeepSeek-R1-Distill-Qwen-1.5B
                       → use DeepSeek-R1-Distill-Qwen-1.5B (same backbone, reasoning-distilled stage)

  For each stage:
    - Bilateral oracle L1: PARAM vs CTX_DEP, N=200/class, pool=10000
    - L2 CC/CW: entropy-matched, N=100/class if sufficient items available
    - Fisher+PCA64 at Layer 26, step-1
    - Entropy baseline

  Note on Stage 1 (base):
    Qwen2.5-1.5B-Base requires instruction wrapping OR raw QA prompts.
    C026 established that Pythia base models produce CTX_DEP=0 (no instruction following).
    C030 (EXPLORATORY) found Qwen2.5-1.5B-Base DOES support bilateral oracle (CTX_DEP=50 from 232 scanned),
    but Fisher near-shuffled at N=50/class. V2 tests at N=200/class with no-framing prompt style.
    For base model: use raw "Q: {question}\nA:" format without chat template.

  Shuffled control for each stage.
  Bootstrap CI (n=1000) for each AUROC.

HYPOTHESES (five competing, no pre-selection — from research_plan_v2.md §8):
  H_IB:  Observability peaks at mid-training (Information Bottleneck) → INVERTED_U
  H_RO:  Observability grows monotonically with routing specialization → MONOTONE_RISE
  H_PC:  Observability shows rapid rise then plateau (Predictive Coding) → PHASE_TRANSITION
  H_AD:  Observability fixed by architecture, invariant to training → FLAT
  H_RD:  Reasoning distillation amplifies observability above SFT → REASONING_JUMP

VERDICT CRITERIA:
  FLAT:              |max_auroc - min_auroc| < 0.05
  MONOTONE_RISE:     base < instruct < reasoning, each step > 0.03
  INVERTED_U:        instruct > max(base, reasoning) by > 0.05
  REASONING_JUMP:    reasoning > instruct by > 0.05, instruct ≈ base
  PHASE_TRANSITION:  large jump base→instruct, plateau instruct→reasoning

GPU: T4 (~12h for 3 stages × ~4h each)
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

STAGES = [
    {
        "name":     "BASE",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "is_instruct": False,
        "layer_idx": 26,
    },
    {
        "name":     "INSTRUCT",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "is_instruct": True,
        "layer_idx": 26,
    },
    {
        "name":     "REASONING",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "is_instruct": True,
        "layer_idx": 26,
    },
]

N_L1_TARGET     = 200
N_L2_TARGET     = 100
POOL_SIZE       = 10_000
N_BOOTSTRAP     = 1000
TRAIN_FRAC      = 0.75
PCA_DIM         = 64
MAX_NEW         = 60
PARAM_MIN_F1    = 0.50
CTX_MAX_F1_NC   = 0.05
CTX_MIN_F1_WC   = 0.50
CW_MAX_F1       = 0.05
ENT_HALF        = 0.30

SAVE_PATH       = "/kaggle/working/exp_l_stage_sweep_v2_results.json"
INTERMEDIATE    = "/kaggle/working/exp_l_stage_sweep_v2_intermediate.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


def _get_hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v: return v
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
        ctx_list = (row.get("entity_pages") or {}).get("wiki_context") or [""]
        ctx = ctx_list[0][:1000] if ctx_list and ctx_list[0] else ""
        items.append({
            "question": row["question"],
            "answers":  row["answer"]["aliases"],
            "context":  ctx,
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
def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {model_id} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    cfg = getattr(model.config, 'text_config', model.config)
    print(f"  Loaded: {cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}", flush=True)
    return model, tokenizer


def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Prompt builders ───────────────────────────────────────────────────────────────
def prompt_nc(tokenizer, q: str, is_instruct: bool) -> str:
    if is_instruct:
        try:
            msgs = [{"role": "user", "content": q}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Base model: raw QA format
    return f"Question: {q}\nAnswer:"


def prompt_wc(tokenizer, q: str, ctx: str, is_instruct: bool) -> str:
    content = f"Context: {ctx[:600]}\n\nQuestion: {q}"
    if is_instruct:
        try:
            msgs = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"Context: {ctx[:600]}\n\nQuestion: {q}\nAnswer:"


# ── Hidden state + entropy ────────────────────────────────────────────────────────
def get_hs_and_entropy(model, tokenizer, prompt: str, layer_idx: int):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()[0]

    h = model.model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    h.remove()

    logits = out.logits[0, -1, :].float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
    probs  = torch.softmax(logits, dim=-1)
    ent    = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
    if not np.isfinite(ent):
        ent = 0.0

    return hs_out[0], ent


def generate(model, tokenizer, prompt: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Probe ─────────────────────────────────────────────────────────────────────────
def run_probe(X_train, y_train, X_test, y_test):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    pca = PCA(n_components=min(PCA_DIM, X_train.shape[1], X_train.shape[0]-1))
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xtr, y_train)
    scores = lda.decision_function(Xte)
    auroc  = float(roc_auc_score(y_test, scores))

    n = len(y_test)
    aurocs = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        try:
            aurocs.append(float(roc_auc_score(y_test[idx], scores[idx])))
        except Exception:
            pass
    ci = (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))) if aurocs else (0., 0.)

    y_s = y_test.copy(); np.random.shuffle(y_s)
    try:
        shuf = float(roc_auc_score(y_s, scores))
    except Exception:
        shuf = 0.5

    return auroc, ci, shuf


def balanced_split(pos_hs, neg_hs):
    n = min(len(pos_hs), len(neg_hs))
    X = np.stack(pos_hs[:n] + neg_hs[:n])
    y = np.array([1]*n + [0]*n)
    n_train = int(n * TRAIN_FRAC)
    pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
    ptr = pi[:n_train]; pte = pi[n_train:n]
    ntr = ni[:n_train]; nte = ni[n_train:n]
    X_tr = np.concatenate([X[ptr], X[ntr]]); y_tr = np.concatenate([np.ones(len(ptr)), np.zeros(len(ntr))])
    X_te = np.concatenate([X[pte], X[nte]]); y_te = np.concatenate([np.ones(len(pte)), np.zeros(len(nte))])
    return X_tr, y_tr, X_te, y_te


# ── Per-stage experiment ──────────────────────────────────────────────────────────
def run_stage(stage: dict, pool: list):
    model_id   = stage["model_id"]
    is_inst    = stage["is_instruct"]
    layer_idx  = stage["layer_idx"]
    stage_name = stage["name"]
    print(f"\n{'='*50}", flush=True)
    print(f"STAGE: {stage_name} ({model_id})", flush=True)

    model, tokenizer = load_model(model_id)

    # ── L1: bilateral oracle ──────────────────────────────────────────────────────
    param_hs, ctxdep_hs = [], []
    n_scanned = 0
    for item in pool:
        if len(param_hs) >= N_L1_TARGET and len(ctxdep_hs) >= N_L1_TARGET:
            break
        n_scanned += 1
        q = item["question"]; ans = item["answers"]; ctx = item["context"]

        pnc = prompt_nc(tokenizer, q, is_inst)
        hs, _ = get_hs_and_entropy(model, tokenizer, pnc, layer_idx)
        if hs is None:
            continue

        gen_nc = generate(model, tokenizer, pnc)
        f1_nc  = token_f1(gen_nc, ans)
        ok_nc  = answer_contains(gen_nc, ans) or f1_nc >= PARAM_MIN_F1

        if ok_nc and len(param_hs) < N_L1_TARGET:
            param_hs.append(hs)
        elif not ok_nc and f1_nc <= CTX_MAX_F1_NC and ctx and len(ctxdep_hs) < N_L1_TARGET:
            pwc    = prompt_wc(tokenizer, q, ctx, is_inst)
            gen_wc = generate(model, tokenizer, pwc)
            f1_wc  = token_f1(gen_wc, ans)
            if answer_contains(gen_wc, ans) or f1_wc >= CTX_MIN_F1_WC:
                ctxdep_hs.append(hs)

        if n_scanned % 500 == 0:
            print(f"  L1 scanned={n_scanned} PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)}", flush=True)

    print(f"L1 done: PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)} scanned={n_scanned}", flush=True)

    l1_result = {"n_param": len(param_hs), "n_ctxdep": len(ctxdep_hs), "n_scanned": n_scanned}
    if len(param_hs) >= 10 and len(ctxdep_hs) >= 10:
        X_tr, y_tr, X_te, y_te = balanced_split(param_hs, ctxdep_hs)
        auroc, ci, shuf = run_probe(X_tr, y_tr, X_te, y_te)
        # Entropy baseline
        from sklearn.metrics import roc_auc_score
        all_ents = []
        for item in pool[:min(n_scanned, 500)]:
            _, ent = get_hs_and_entropy(model, tokenizer, prompt_nc(tokenizer, item["question"], is_inst), layer_idx)
            if np.isfinite(ent) and ent > 0:
                all_ents.append(ent)
        l1_result.update({
            "fisher_auroc": auroc, "fisher_ci_95": list(ci), "fisher_shuffled": shuf,
        })
        print(f"L1 Fisher={auroc:.4f} CI=[{ci[0]:.3f},{ci[1]:.3f}] shuf={shuf:.4f}", flush=True)
    else:
        l1_result["skip_reason"] = f"Insufficient data for probe"
        print(f"L1 SKIPPED: insufficient data", flush=True)

    # ── L2: CC/CW entropy-matched ─────────────────────────────────────────────────
    l2_result = {}
    # Calibrate entropy
    sample_ents = []
    for item in pool[:500]:
        _, ent = get_hs_and_entropy(model, tokenizer, prompt_nc(tokenizer, item["question"], is_inst), layer_idx)
        if np.isfinite(ent) and ent > 0:
            sample_ents.append(ent)

    if len(sample_ents) >= 10:
        theta_conf = float(np.percentile(sample_ents, 30))
        ent_lo = theta_conf - ENT_HALF
        ent_hi = theta_conf + ENT_HALF
        print(f"L2 entropy window: [{ent_lo:.4f}, {ent_hi:.4f}]", flush=True)

        cc_hs, cw_hs, cc_ents, cw_ents = [], [], [], []
        for item in pool:
            if len(cc_hs) >= N_L2_TARGET and len(cw_hs) >= N_L2_TARGET:
                break
            q = item["question"]; ans = item["answers"]
            pnc = prompt_nc(tokenizer, q, is_inst)
            hs, ent = get_hs_and_entropy(model, tokenizer, pnc, layer_idx)
            if hs is None or not np.isfinite(ent) or not (ent_lo <= ent <= ent_hi):
                continue
            gen_nc = generate(model, tokenizer, pnc)
            f1 = token_f1(gen_nc, ans)
            ok = answer_contains(gen_nc, ans) or f1 >= PARAM_MIN_F1
            if ok and len(cc_hs) < N_L2_TARGET:
                cc_hs.append(hs); cc_ents.append(ent)
            elif f1 <= CW_MAX_F1 and len(cw_hs) < N_L2_TARGET:
                cw_hs.append(hs); cw_ents.append(ent)

        print(f"L2 collected: CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
        if len(cc_hs) >= 10 and len(cw_hs) >= 10:
            X_tr, y_tr, X_te, y_te = balanced_split(cc_hs, cw_hs)
            fisher_auroc, fisher_ci, fisher_shuf = run_probe(X_tr, y_tr, X_te, y_te)
            from sklearn.metrics import roc_auc_score
            n2 = min(len(cc_hs), len(cw_hs))
            ents_all = np.array(cc_ents[:n2] + cw_ents[:n2])
            labels2  = np.array([1]*n2 + [0]*n2)
            try:
                ent_auroc = float(roc_auc_score(labels2, -ents_all))
            except Exception:
                ent_auroc = 0.5
            gap = fisher_auroc - ent_auroc
            print(f"L2 Fisher={fisher_auroc:.4f} Entropy={ent_auroc:.4f} Gap={gap:.4f}", flush=True)
            l2_result = {
                "n_cc": len(cc_hs), "n_cw": len(cw_hs),
                "theta_conf": theta_conf, "ent_window": [ent_lo, ent_hi],
                "fisher_auroc": fisher_auroc, "fisher_ci_95": list(fisher_ci),
                "fisher_shuffled": fisher_shuf,
                "entropy_auroc": ent_auroc, "gap": gap,
                "l2_verdict": "SUPPORTED" if fisher_auroc >= 0.70 and gap >= 0.10 else "NOT_SUPPORTED",
            }
        else:
            l2_result = {"skip_reason": f"Insufficient items in entropy window: CC={len(cc_hs)} CW={len(cw_hs)}"}
    else:
        l2_result = {"skip_reason": "All entropy values NaN/zero — base model may not produce meaningful entropy"}

    unload_model(model)
    return {"stage": stage_name, "model_id": model_id, "l1": l1_result, "l2": l2_result}


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    stage_results = []

    for stage in STAGES:
        result = run_stage(stage, pool)
        stage_results.append(result)

        # Intermediate save
        try:
            with open(INTERMEDIATE, "w") as f:
                json.dump({"stages_done": len(stage_results), "results": stage_results}, f, indent=2)
        except Exception:
            pass

    # Verdict: which hypothesis?
    l1_aurocs = [r["l1"].get("fisher_auroc") for r in stage_results if r["l1"].get("fisher_auroc")]
    if len(l1_aurocs) == 3:
        b, i, r_ = l1_aurocs
        spread = max(l1_aurocs) - min(l1_aurocs)
        if spread < 0.05:
            verdict = "FLAT"
        elif b < i < r_ and (i - b) > 0.03 and (r_ - i) > 0.03:
            verdict = "MONOTONE_RISE"
        elif i > max(b, r_) + 0.05:
            verdict = "INVERTED_U"
        elif r_ > i + 0.05 and abs(i - b) < 0.05:
            verdict = "REASONING_JUMP"
        else:
            verdict = "PHASE_TRANSITION_OR_UNCLEAR"
    else:
        verdict = "INCOMPLETE"

    print(f"\n{'='*50}", flush=True)
    print(f"STAGE SWEEP VERDICT: {verdict}", flush=True)
    if l1_aurocs:
        for name, val in zip(["BASE", "INSTRUCT", "REASONING"], l1_aurocs):
            print(f"  {name}: L1={val:.4f}", flush=True)

    results = {
        "experiment":   "EXP_L_STAGE_SWEEP_V2",
        "n_l1_target":  N_L1_TARGET,
        "n_l2_target":  N_L2_TARGET,
        "pool_size":    POOL_SIZE,
        "stages":       stage_results,
        "l1_aurocs":    l1_aurocs,
        "verdict":      verdict,
        "elapsed_min":  (time.time() - t0) / 60,
        "hypotheses": {
            "FLAT":                   "H_AD (Architectural Determination): training does not change observability",
            "MONOTONE_RISE":          "H_RO (Routing Optimization): observability grows with specialization",
            "INVERTED_U":             "H_IB (Information Bottleneck): peak at SFT/RLHF stage",
            "REASONING_JUMP":         "H_RD (Reasoning Distillation): geometry sharpened by reasoning training",
            "PHASE_TRANSITION_OR_UNCLEAR": "H_PC (Predictive Coding) or multi-factor — additional experiments needed",
        },
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)
    print(f"Elapsed: {results['elapsed_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()

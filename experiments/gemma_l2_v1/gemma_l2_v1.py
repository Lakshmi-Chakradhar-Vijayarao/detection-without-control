#!/usr/bin/env python3
"""
gemma_l2_v1.py — EXP_GEMMA_L2_V1
Phase 2 (L2 False Certainty) only for Gemma-2-2B-IT.

Phase 1 (L1 bilateral oracle) is complete:
  AUROC=0.7528, shuffled=0.5296 (CLEAN), CI=[0.652, 0.848], N=200/class

WHY THIS SCRIPT EXISTS:
  gemma_bilateral_v1.py Phase 2 produced theta_conf=nan because Gemma-2 applies
  final_logit_softcapping (tanh(x/30)*30). In float16, extreme pre-softcap
  activations produce NaN logits for some vocab positions. torch.softmax
  propagates NaN → entropy=NaN → np.percentile([NaN,...])=NaN → every entropy
  window comparison fails → CC=0, CW=0.

  Fix: cast logits to float32 before softmax. Filter NaN from calibration sample.

PHASE 1 CONSTANTS (hardcoded from gemma_bilateral_v1 run):
  l1_auroc    = 0.7528
  l1_shuffled = 0.5296
  l1_ci_95    = [0.652, 0.848]
  n_param     = 200
  n_ctxdep    = 200
  n_scanned   = 2342
  layer_idx   = 24  (penultimate of 26 layers)

DESIGN (Phase 2):
  Entropy-matched CC/CW collection with float32 entropy.
  N=100/class target. Pool=15000 (full pool, fresh shuffle).
  Fisher+PCA64 vs entropy gap. BO_Transfer from hardcoded Phase 1 probe
  is NOT available (no saved probe weights) — will be skipped.

VERDICT: SUPPORTED if Fisher AUROC >= 0.70 AND gap >= 0.10
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID        = "google/gemma-2-2b-it"
LAYER_IDX       = 24          # penultimate of 26 layers
N_CC_TARGET     = 100
N_CW_TARGET     = 100
POOL_SIZE       = 15_000
N_BOOTSTRAP     = 1000
TRAIN_FRAC      = 0.75
PCA_DIM         = 64
MAX_NEW         = 60
PARAM_MIN_F1    = 0.50
CW_MAX_F1       = 0.05
ENT_HALF        = 0.35
CALIB_SAMPLE    = 500

# Phase 1 results (hardcoded)
PHASE1 = {
    "l1_auroc":    0.7528,
    "l1_shuffled": 0.5296,
    "l1_ci_95":    [0.652, 0.848],
    "n_param":     200,
    "n_ctxdep":    200,
    "n_scanned":   2342,
}

SAVE_PATH = "/kaggle/working/gemma_l2_v1_results.json"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"Pool loaded: {len(items)} items", flush=True)
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
    print(f"  Layers: {cfg.num_hidden_layers}, hidden: {cfg.hidden_size}", flush=True)
    return model, tokenizer


_LAYER_PATHS = ["model.layers", "model.language_model.layers",
                "language_model.model.layers", "transformer.h"]

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
    raise RuntimeError("Cannot find transformer layers")


def prompt_nc(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_hs_and_entropy(model, tokenizer, prompt: str):
    """
    Returns (hidden_state, entropy) at layer LAYER_IDX, step-1.
    KEY FIX: cast logits to float32 before softmax to avoid Gemma-2
    NaN from final_logit_softcapping in float16.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()[0]

    layers = get_layers(model)
    h = layers[LAYER_IDX].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    h.remove()

    # float32 cast is the fix
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


# ── Calibrate entropy threshold ───────────────────────────────────────────────────
def calibrate_theta(model, tokenizer, pool):
    print("Calibrating entropy threshold …", flush=True)
    sample_ents = []
    for item in pool[:CALIB_SAMPLE]:
        _, ent = get_hs_and_entropy(model, tokenizer, prompt_nc(tokenizer, item["question"]))
        if np.isfinite(ent) and ent > 0:
            sample_ents.append(ent)
    if not sample_ents:
        raise RuntimeError("All entropy values NaN/zero during calibration. Check model logits.")
    finite_frac = len(sample_ents) / CALIB_SAMPLE
    theta_conf  = float(np.percentile(sample_ents, 30))
    ent_lo      = theta_conf - ENT_HALF
    ent_hi      = theta_conf + ENT_HALF
    print(f"  sample_ents: {len(sample_ents)}/{CALIB_SAMPLE} finite ({finite_frac:.1%})", flush=True)
    print(f"  theta_conf={theta_conf:.4f}  window=[{ent_lo:.4f}, {ent_hi:.4f}]", flush=True)
    return theta_conf, ent_lo, ent_hi


# ── Collect CC/CW ─────────────────────────────────────────────────────────────────
def collect_l2(model, tokenizer, pool, ent_lo, ent_hi):
    print("\nCollecting CC/CW items …", flush=True)
    cc_hs, cw_hs = [], []
    cc_ents, cw_ents = [], []
    n_scanned = 0

    for item in pool:
        if len(cc_hs) >= N_CC_TARGET and len(cw_hs) >= N_CW_TARGET:
            break
        n_scanned += 1
        q   = item["question"]
        ans = item["answers"]

        pnc = prompt_nc(tokenizer, q)
        hs, ent = get_hs_and_entropy(model, tokenizer, pnc)
        if hs is None or not np.isfinite(ent) or not (ent_lo <= ent <= ent_hi):
            continue

        gen_nc = generate(model, tokenizer, pnc)
        f1     = token_f1(gen_nc, ans)
        ok     = answer_contains(gen_nc, ans) or f1 >= PARAM_MIN_F1

        if ok and len(cc_hs) < N_CC_TARGET:
            cc_hs.append(hs); cc_ents.append(ent)
        elif f1 <= CW_MAX_F1 and len(cw_hs) < N_CW_TARGET:
            cw_hs.append(hs); cw_ents.append(ent)

        if n_scanned % 1000 == 0:
            print(f"  scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)

    print(f"Collection done: scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
    return cc_hs, cw_hs, cc_ents, cw_ents, n_scanned


# ── Probe ─────────────────────────────────────────────────────────────────────────
def run_probe(cc_hs, cw_hs, cc_ents, cw_ents):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    n = min(len(cc_hs), len(cw_hs))
    if n < 10:
        return {"l2_skip": True, "l2_reason": f"Insufficient items: CC={len(cc_hs)} CW={len(cw_hs)}"}

    X    = np.stack(cc_hs[:n] + cw_hs[:n])
    y    = np.array([1]*n + [0]*n)
    ents = np.array(cc_ents[:n] + cw_ents[:n])

    n_train = int(n * TRAIN_FRAC)
    cc_idx = np.where(y == 1)[0]; cw_idx = np.where(y == 0)[0]
    ptr = cc_idx[:n_train]; pte = cc_idx[n_train:n]
    ctr = cw_idx[:n_train]; cte = cw_idx[n_train:n]
    X_train = np.concatenate([X[ptr], X[ctr]])
    y_train = np.concatenate([np.ones(len(ptr)), np.zeros(len(ctr))])
    X_test  = np.concatenate([X[pte], X[cte]])
    y_test  = np.concatenate([np.ones(len(pte)), np.zeros(len(cte))])

    # Fisher AUROC
    pca = PCA(n_components=min(PCA_DIM, X_train.shape[1], X_train.shape[0]-1))
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xtr, y_train)
    scores = lda.decision_function(Xte)
    fisher_auroc = float(roc_auc_score(y_test, scores))

    # Bootstrap CI
    aurocs = []
    n_te = len(y_test)
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n_te, n_te, replace=True)
        try:
            aurocs.append(float(roc_auc_score(y_test[idx], scores[idx])))
        except Exception:
            pass
    ci = (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))) if aurocs else (0., 0.)

    # Shuffled
    y_s = y_test.copy(); np.random.shuffle(y_s)
    try:
        shuf = float(roc_auc_score(y_s, scores))
    except Exception:
        shuf = 0.5

    # Entropy AUROC
    ents_te = ents[np.concatenate([pte, cte])]
    try:
        ent_auroc = float(roc_auc_score(y_test, -ents_te))
    except Exception:
        ent_auroc = 0.5

    gap     = fisher_auroc - ent_auroc
    verdict = "SUPPORTED" if fisher_auroc >= 0.70 and gap >= 0.10 else "NOT_SUPPORTED"

    print(f"\nL2 Fisher={fisher_auroc:.4f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  shuf={shuf:.4f}", flush=True)
    print(f"L2 Entropy={ent_auroc:.4f}  Gap={gap:.4f}", flush=True)
    print(f"L2 VERDICT: {verdict}", flush=True)

    return {
        "n_cc": n, "n_cw": n,
        "fisher_auroc": fisher_auroc, "fisher_ci_95": list(ci),
        "fisher_shuffled": shuf,
        "entropy_auroc": ent_auroc,
        "gap": gap,
        "cc_ent_mean": float(np.mean(cc_ents[:n])),
        "cw_ent_mean": float(np.mean(cw_ents[:n])),
        "l2_verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer = load_model()

    theta_conf, ent_lo, ent_hi = calibrate_theta(model, tokenizer, pool)
    cc_hs, cw_hs, cc_ents, cw_ents, n_scanned_l2 = collect_l2(
        model, tokenizer, pool, ent_lo, ent_hi
    )
    l2 = run_probe(cc_hs, cw_hs, cc_ents, cw_ents)

    results = {
        "experiment":   "EXP_GEMMA_L2_V1",
        "model":        MODEL_ID,
        "layer_idx":    LAYER_IDX,
        "phase1":       PHASE1,
        "l2_calib":     {"theta_conf": theta_conf, "ent_lo": ent_lo, "ent_hi": ent_hi,
                         "n_scanned": n_scanned_l2},
        "l2":           l2,
        "elapsed_min":  (time.time() - t0) / 60,
        "bug_fixed":    "float32 logit cast + nan_to_num before softmax for Gemma-2 final_logit_softcapping",
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)
    print(f"Elapsed: {results['elapsed_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()

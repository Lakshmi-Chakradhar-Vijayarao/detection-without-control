#!/usr/bin/env python3
"""
co_gemma_mistral_v1.py — EXP_CO_GEMMA_MISTRAL_V1

SCIENTIFIC QUESTION:
  C036 resolution: Is T2_L2 a measurement framing failure or geometry absence?

  Gemma (theta_conf=0.067) and Mistral (theta_conf=0.1217) both failed
  entropy-matched L2 (gap=0.022 and gap=-0.014 respectively). But both
  show BO_Transfer > 0.5 — hidden geometry EXISTS.

  Hypothesis: entropy-matched framing selects too narrow a window when
  theta_conf < 0.15, yielding insufficient items and noise-dominated results.
  CO labeling (no entropy window, just entropy ≤ theta_conf) should recover
  the Fisher gap.

DESIGN:
  For each model (Gemma-2-2B-IT, then Mistral-7B-Instruct-v0.3):
    Phase 1 — Calibrate theta_conf (30th percentile of N=800 sample)
    Phase 2 — CO-L2 collection (N=200/class):
      CC: entropy ≤ theta_conf AND (answer_contains OR token_f1 ≥ 0.50)
      CW: entropy ≤ theta_conf AND token_f1 ≤ 0.05
      No entropy window — just the confident filter.
    Phase 3 — Fisher+PCA64 probe + shuffled control + bootstrap CI

VERDICTS:
  CO_RECOVERS: Fisher AUROC ≥ 0.65 (gap ≥ 0.15 over 0.50 baseline)
  CO_PARTIAL: 0.55 ≤ AUROC < 0.65
  CO_FAILS: AUROC < 0.55

  Overall:
    C036_CONFIRMED: both models CO_RECOVERS → T2_L2 = framing failure
    C036_MIXED: one model recovers → architecture-dependent framing effect
    C036_FALSIFIED: neither recovers → geometry genuinely absent

GPU: T4 (~10-14h, both models sequential)
Models: google/gemma-2-2b-it, mistralai/Mistral-7B-Instruct-v0.3
Dataset: TriviaQA rc.wikipedia train
"""

from __future__ import annotations
import subprocess as _sp
# bitsandbytes required for Mistral-7B 8-bit quantization on T4
_sp.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
GEMMA_MODEL_ID   = "google/gemma-2-2b-it"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
N_TARGET         = 200          # per-class, per model
POOL_SIZE        = 20_000
N_CALIB          = 800          # calibration sample for theta_conf
THETA_PERCENTILE = 30           # percentile for theta_conf
N_BOOTSTRAP      = 1000
TRAIN_FRAC       = 0.75
PCA_DIM          = 64
MAX_NEW          = 60
PARAM_MIN_F1     = 0.50
CW_MAX_F1        = 0.05

_BRIEF = "Answer the following question in one short phrase or name only. Do not explain."

RESULTS_FILE   = "/kaggle/working/co_gemma_mistral_v1_results.json"
INTERMEDIATE   = "/kaggle/working/co_gemma_mistral_v1_intermediate.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required — no CUDA device found.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

t0_global = time.time()


# ── HF token ─────────────────────────────────────────────────────────────────
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

_tok_val = _get_hf_token()
if _tok_val:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_tok_val, add_to_git_credential=False)
    print("HF login: OK", flush=True)
else:
    print("WARNING: HF_TOKEN not found", flush=True)


# ── Data ─────────────────────────────────────────────────────────────────────
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
    print(f"Loaded pool: {len(items)} items", flush=True)
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


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_id: str, use_8bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {model_id} (8bit={use_8bit}) …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True
        ).to(DEVICE)
    model.eval()
    cfg = getattr(model.config, 'text_config', model.config)
    n_layers = cfg.num_hidden_layers
    hidden   = cfg.hidden_size
    layer_idx = n_layers - 2   # penultimate
    print(f"  n_layers={n_layers}, hidden={hidden}, probe_layer={layer_idx}", flush=True)
    return model, tokenizer, layer_idx


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(5)
    if torch.cuda.is_available():
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"Model unloaded. VRAM free: {free/1e9:.1f} GB", flush=True)
    else:
        print("Model unloaded.", flush=True)


# ── Layer resolver ────────────────────────────────────────────────────────────
_LAYER_PATHS = [
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "transformer.h",
]

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
    raise RuntimeError(f"Cannot find transformer layers in {type(model).__name__}")


# ── Hidden state + entropy ────────────────────────────────────────────────────
def get_hs_and_entropy(model, tokenizer, prompt: str, layer_idx: int,
                       is_gemma: bool = False):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()

    layers = get_layers(model)
    h = layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    h.remove()

    logits = out.logits[0, -1, :]
    if is_gemma:
        # float16 + final_logit_softcapping (tanh(x/30)×30) causes NaN
        logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
    probs  = torch.softmax(logits, dim=-1)
    ent    = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
    if not np.isfinite(ent) or ent < 0:
        ent = 0.0

    hs = hs_out[0][0] if hs_out[0] is not None else None
    return hs, ent


def generate(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Prompts ───────────────────────────────────────────────────────────────────
def prompt_nc(tokenizer, q: str, is_gemma: bool = False) -> str:
    # Gemma-2 chat template does not support the system role — embed in user turn
    if is_gemma:
        msgs = [{"role": "user", "content": f"{_BRIEF}\n\nQuestion: {q}"}]
    else:
        msgs = [
            {"role": "system", "content": _BRIEF},
            {"role": "user",   "content": q},
        ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Probe ─────────────────────────────────────────────────────────────────────
def fit_and_eval(X: np.ndarray, y: np.ndarray):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    n_min = min(np.sum(y == 0), np.sum(y == 1))
    n_train = int(n_min * TRAIN_FRAC)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    tr = np.concatenate([idx0[:n_train], idx1[:n_train]])
    te = np.concatenate([idx0[n_train:n_min], idx1[n_train:n_min]])

    pca = PCA(n_components=min(PCA_DIM, X.shape[1], len(tr)-1))
    Xp_tr = pca.fit_transform(X[tr])
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xp_tr, y[tr])

    Xp_te = pca.transform(X[te])
    scores = lda.decision_function(Xp_te)
    auroc = float(roc_auc_score(y[te], scores))

    y_shuf = y[te].copy(); np.random.shuffle(y_shuf)
    try:
        shuf = float(roc_auc_score(y_shuf, scores))
    except Exception:
        shuf = 0.5

    aurocs = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(len(te), len(te), replace=True)
        try:
            aurocs.append(float(roc_auc_score(y[te][idx], scores[idx])))
        except Exception:
            pass
    ci = (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))) \
        if aurocs else (0.0, 0.0)

    return auroc, shuf, ci


def entropy_auroc(ents: list, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, -np.array(ents)))
    except Exception:
        return 0.5


# ── Per-model experiment ──────────────────────────────────────────────────────
def run_model(model_id: str, pool: list, is_gemma: bool) -> dict:
    label = "gemma" if is_gemma else "mistral"
    print(f"\n{'='*60}", flush=True)
    print(f"Running CO-L2 on {model_id}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()

    # Mistral-7B in float16 ≈ 14 GB — too large for T4 (15.6 GB) after Gemma unload
    use_8bit = not is_gemma
    model, tokenizer, layer_idx = load_model(model_id, use_8bit=use_8bit)

    # Phase 1: calibrate theta_conf
    print(f"\n--- Phase 1: Calibrating theta_conf (N={N_CALIB} sample) ---", flush=True)
    calib_ents = []
    for item in pool[:N_CALIB]:
        pnc = prompt_nc(tokenizer, item["question"], is_gemma)
        _, ent = get_hs_and_entropy(model, tokenizer, pnc, layer_idx, is_gemma)
        if np.isfinite(ent) and ent > 0:
            calib_ents.append(ent)
    if len(calib_ents) < 100:
        raise RuntimeError(f"[{label}] Too few finite entropy values: {len(calib_ents)}")
    theta_conf = float(np.percentile(calib_ents, THETA_PERCENTILE))
    print(f"  theta_conf={theta_conf:.4f} ({THETA_PERCENTILE}th pctile of {len(calib_ents)} samples)", flush=True)

    # Phase 2: CO-L2 collection
    print(f"\n--- Phase 2: CO-L2 collection (N={N_TARGET}/class) ---", flush=True)
    print(f"  Confident filter: entropy ≤ {theta_conf:.4f} (no window matching)", flush=True)
    cc_hs, cc_ents = [], []
    cw_hs, cw_ents = [], []
    n_scanned = 0

    for item in pool[N_CALIB:]:
        if len(cc_hs) >= N_TARGET and len(cw_hs) >= N_TARGET:
            break
        n_scanned += 1
        q   = item["question"]
        ans = item["answers"]

        pnc = prompt_nc(tokenizer, q, is_gemma)
        hs, ent = get_hs_and_entropy(model, tokenizer, pnc, layer_idx, is_gemma)
        if hs is None or ent > theta_conf:
            continue

        gen  = generate(model, tokenizer, pnc)
        f1   = token_f1(gen, ans)
        ok   = answer_contains(gen, ans) or f1 >= PARAM_MIN_F1

        if ok and len(cc_hs) < N_TARGET:
            cc_hs.append(hs); cc_ents.append(ent)
        elif f1 <= CW_MAX_F1 and len(cw_hs) < N_TARGET:
            cw_hs.append(hs); cw_ents.append(ent)

        if n_scanned % 200 == 0:
            print(f"  scanned={n_scanned} CC={len(cc_hs)} CW={len(cw_hs)} "
                  f"(elapsed: {(time.time()-t0)/60:.0f}m)", flush=True)

        if (len(cc_hs) + len(cw_hs)) % 50 == 0 and (len(cc_hs) + len(cw_hs)) > 0:
            try:
                with open(INTERMEDIATE, "w") as f:
                    json.dump({
                        "model": label, "phase": "co_l2",
                        "cc": len(cc_hs), "cw": len(cw_hs),
                        "scanned": n_scanned,
                    }, f)
            except Exception:
                pass

    print(f"\nCO-L2 collection done: CC={len(cc_hs)} CW={len(cw_hs)}", flush=True)
    if len(cc_hs) < 50 or len(cw_hs) < 50:
        raise RuntimeError(f"[{label}] Insufficient CO-L2 data: CC={len(cc_hs)} CW={len(cw_hs)}")

    # Phase 3: Fisher probe
    print(f"\n--- Phase 3: Fisher+PCA64 probe ---", flush=True)
    n_use = min(len(cc_hs), len(cw_hs))
    X = np.stack(cc_hs[:n_use] + cw_hs[:n_use])
    y = np.array([1]*n_use + [0]*n_use)

    fisher_auroc, shuf_auroc, ci = fit_and_eval(X, y)
    ents_combined = cc_ents[:n_use] + cw_ents[:n_use]
    ent_auc = entropy_auroc(ents_combined, y)
    gap = fisher_auroc - ent_auc

    shuf_status = "CLEAN" if shuf_auroc < 0.62 else ("WARN" if shuf_auroc < 0.70 else "FAIL")

    if fisher_auroc >= 0.65:
        verdict = "CO_RECOVERS"
    elif fisher_auroc >= 0.55:
        verdict = "CO_PARTIAL"
    else:
        verdict = "CO_FAILS"

    elapsed = (time.time() - t0) / 60
    print(f"\n[{label.upper()}] Results:", flush=True)
    print(f"  Fisher AUROC = {fisher_auroc:.4f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]", flush=True)
    print(f"  Entropy AUROC = {ent_auc:.4f}  Gap = {gap:+.4f}", flush=True)
    print(f"  Shuffled = {shuf_auroc:.4f} ({shuf_status})", flush=True)
    print(f"  theta_conf = {theta_conf:.4f}  N_CC={n_use}  N_CW={n_use}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  Elapsed: {elapsed:.0f} min", flush=True)

    result = {
        "model": model_id,
        "theta_conf": theta_conf,
        "n_cc": n_use,
        "n_cw": n_use,
        "n_scanned": n_scanned,
        "fisher_auroc": fisher_auroc,
        "entropy_auroc": ent_auc,
        "gap": gap,
        "shuffled_auroc": shuf_auroc,
        "shuffled_status": shuf_status,
        "ci_95": list(ci),
        "verdict": verdict,
        "elapsed_min": elapsed,
    }

    unload_model(model)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n=== EXP_CO_GEMMA_MISTRAL_V1 ===", flush=True)
    print("Question: Does CO labeling recover L2 for low-theta_conf architectures?", flush=True)
    print("C036 test: T2_L2 = framing failure vs geometry absence", flush=True)
    print(f"Models: {GEMMA_MODEL_ID}, {MISTRAL_MODEL_ID}", flush=True)
    print(f"N={N_TARGET}/class per model, theta_conf=30th pctile", flush=True)

    pool = load_pool(POOL_SIZE)

    results = {}

    # --- Gemma ---
    try:
        results["gemma"] = run_model(GEMMA_MODEL_ID, pool, is_gemma=True)
    except Exception as e:
        print(f"\nERROR on Gemma: {e}", flush=True)
        results["gemma"] = {"error": str(e)}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Mistral ---
    try:
        results["mistral"] = run_model(MISTRAL_MODEL_ID, pool, is_gemma=False)
    except Exception as e:
        print(f"\nERROR on Mistral: {e}", flush=True)
        results["mistral"] = {"error": str(e)}

    # --- Overall verdict ---
    gemma_ok    = results.get("gemma", {}).get("verdict") == "CO_RECOVERS"
    mistral_ok  = results.get("mistral", {}).get("verdict") == "CO_RECOVERS"
    gemma_part  = results.get("gemma", {}).get("verdict") == "CO_PARTIAL"
    mistral_part= results.get("mistral", {}).get("verdict") == "CO_PARTIAL"

    if gemma_ok and mistral_ok:
        overall = "C036_CONFIRMED"
    elif gemma_ok or mistral_ok or gemma_part or mistral_part:
        overall = "C036_MIXED"
    else:
        overall = "C036_FALSIFIED"

    elapsed_total = (time.time() - t0_global) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL VERDICT: {overall}", flush=True)
    print(f"  Gemma:   {results.get('gemma', {}).get('verdict', 'ERROR')} "
          f"(Fisher={results.get('gemma', {}).get('fisher_auroc', 'N/A')})", flush=True)
    print(f"  Mistral: {results.get('mistral', {}).get('verdict', 'ERROR')} "
          f"(Fisher={results.get('mistral', {}).get('fisher_auroc', 'N/A')})", flush=True)
    print(f"Total elapsed: {elapsed_total:.0f} min", flush=True)

    final = {
        "experiment": "co_gemma_mistral_v1",
        "overall_verdict": overall,
        "gemma": results.get("gemma", {}),
        "mistral": results.get("mistral", {}),
        "total_elapsed_min": elapsed_total,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)

    return final


if __name__ == "__main__":
    main()

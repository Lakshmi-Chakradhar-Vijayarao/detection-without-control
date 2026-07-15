"""
experiments/rlhf_attenuation_yi_v1/rlhf_attenuation_yi_v1.py

RLHF ATTENUATION — Yi-6B (GQA)
================================

Third GQA family experiment. OLMo-7B was the original target but OLMo-7B-hf ignores
BitsAndBytes quantization config (loads full fp16, ~14 GB, OOM on T4). Yi-6B is a
clean alternative: genuine GQA, 6B params, standard HF format, different organization
(01-ai), no trust_remote_code issues.

Models:
  Base:     01-ai/Yi-6B
  Instruct: 01-ai/Yi-6B-Chat

Prior GQA results:
  Qwen2.5-1.5B (Alibaba): base=0.899  inst=0.864  Δ=-0.036
  Llama-3.2-3B (Meta):    base=0.665  inst=0.629  Δ=-0.036

Hypothesis: Δ ≈ -0.036 (third independent GQA family confirmation).
If Yi also shows Δ=-0.036: GQA universality holds across organizations.

Memory notes:
  Yi-6B in 4-bit ≈ 3 GB (comfortably within T4's 15 GB).
  Using device_map="auto" to prevent fp16 intermediate materialization.

GPU: T4 (sm_75).
"""

from __future__ import annotations
import gc, json, os, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes>=0.46.1", "scikit-learn", "datasets", "huggingface_hub"],
               check=False)

import numpy as np
import torch

# ── HF login ──────────────────────────────────────────────────────────────────
try:
    _tok = ""
    try:
        from kaggle_secrets import UserSecretsClient as _USC
        _tok = _USC().get_secret("HF_TOKEN")
    except Exception:
        pass
    _tok = _tok or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if _tok:
        from huggingface_hub import login as _login
        _login(token=_tok, add_to_git_credential=False)
        print("HF login: OK", flush=True)
except Exception as e:
    print(f"HF login: {e}", flush=True)

assert torch.cuda.is_available(), "GPU required"
DEVICE = "cuda"
_sm = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{_sm[0]*10+_sm[1]}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_CAL   = 30
MAX_GEN = 150
SEED    = 42
np.random.seed(SEED)

BASE_ID = "01-ai/Yi-6B"
INST_ID = "01-ai/Yi-6B-Chat"
ARCH    = "GQA"

OUT_DIR  = Path("/kaggle/working")
OUT_FILE = OUT_DIR / "rlhf_attenuation_yi_v1_results.json"

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(n: int = 300) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    pool = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        pool.append({"question": row["question"], "answers": row["answer"]["aliases"]})
        if len(pool) >= n:
            break
    print(f"Loaded {len(pool)} TriviaQA questions", flush=True)
    return pool

def answer_contains(pred: str, golds: List[str]) -> bool:
    p = pred.lower()
    return any(g.lower().strip() in p for g in golds if g.strip())

def token_f1(pred: str, golds: List[str]) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        c = p & q
        if not c or not p or not q:
            continue
        pr, rc = len(c)/len(p), len(c)/len(q)
        best = max(best, 2*pr*rc/(pr+rc))
    return best

# ── Model loading ─────────────────────────────────────────────────────────────
_LAYER_PATHS = [
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "transformer.h",
    "model.transformer.blocks",
]

def get_layers(model):
    for path in _LAYER_PATHS:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                print(f"  [layers] {path} ({len(obj)} layers)", flush=True)
                return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot find layers in {type(model).__name__}")

def _input_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return DEVICE

def load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {model_id} …", flush=True)
    print(f"  VRAM before load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
    ).eval()
    n = mdl.config.num_hidden_layers
    print(f"  n_layers={n}  hidden={mdl.config.hidden_size}", flush=True)
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    return mdl, tok

def release_model(mdl):
    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  VRAM after release: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# ── Gen-step-1 hidden state extraction ────────────────────────────────────────
def get_step1_hs(model, tok, prompt: str, layer_idx: int) -> Optional[np.ndarray]:
    dev = _input_device(model)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)
    captured = [None]
    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] == 1:
            captured[0] = x[0, -1, :].float().detach().cpu().numpy()
    h = get_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            pre = model(ids, use_cache=True)
            model(ids[:, -1:], past_key_values=pre.past_key_values, use_cache=False)
    finally:
        h.remove()
    return captured[0]

# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate(model, tok, data: List[Dict], layer_idx: int,
              model_id: str) -> Tuple[np.ndarray, float, float, float]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    print(f"\n  Calibrating {model_id} — layer {layer_idx}", flush=True)
    param_hs, wrong_hs = [], []
    shuffled = list(data)
    np.random.shuffle(shuffled)
    t0 = time.time()

    for i, s in enumerate(shuffled):
        if len(param_hs) >= N_CAL and len(wrong_hs) >= N_CAL:
            print(f"    Done at sample {i}  elapsed={time.time()-t0:.0f}s", flush=True)
            break
        if i % 10 == 0:
            print(f"    [{i}] PARAM={len(param_hs)} WRONG={len(wrong_hs)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

        # Yi base uses completion format; Yi-Chat uses chat template
        if "Chat" in model_id or "chat" in model_id or "Instruct" in model_id:
            try:
                msgs = [{"role": "user", "content": s["question"]}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"Question: {s['question']}\nAnswer:"
        else:
            prompt = f"Question: {s['question']}\nAnswer:"

        dev = _input_device(model)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_GEN, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        correct = answer_contains(gen, s["answers"]) or token_f1(gen, s["answers"]) >= 0.4

        if correct and len(param_hs) >= N_CAL:
            continue
        if not correct and len(wrong_hs) >= N_CAL:
            continue

        hs = get_step1_hs(model, tok, prompt, layer_idx)
        if hs is None:
            continue
        if correct:
            param_hs.append(hs)
        else:
            wrong_hs.append(hs)

    np_, nw = len(param_hs), len(wrong_hs)
    print(f"  Calibration: PARAM={np_} WRONG={nw}  elapsed={time.time()-t0:.0f}s", flush=True)
    if np_ < 5 or nw < 5:
        raise RuntimeError(f"Insufficient calibration samples: PARAM={np_}, WRONG={nw}")

    X = np.stack(param_hs + wrong_hs)
    y = np.array([1]*np_ + [0]*nw)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_aurocs = []
    for tr_idx, val_idx in cv.split(X, y):
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X[tr_idx], y[tr_idx])
        scores = lda.decision_function(X[val_idx])
        fold_aurocs.append(roc_auc_score(y[val_idx], scores))
    cv_auroc = float(np.mean(fold_aurocs))

    lda_full = LinearDiscriminantAnalysis(n_components=1)
    lda_full.fit(X, y)
    d = lda_full.coef_[0] / (np.linalg.norm(lda_full.coef_[0]) + 1e-9)
    projs = X @ d
    mu_p = float(np.mean(projs[y == 1]))
    mu_w = float(np.mean(projs[y == 0]))

    print(f"  CV AUROC (3-fold): {cv_auroc:.4f}  mu_PARAM={mu_p:.3f}  mu_WRONG={mu_w:.3f}",
          flush=True)
    return d, mu_p, mu_w, cv_auroc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"RLHF Attenuation — Yi-6B (GQA)  |  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Base: {BASE_ID}", flush=True)
    print(f"Inst: {INST_ID}", flush=True)
    print(f"Prior GQA Δ: Qwen2.5=-0.036, Llama=-0.036", flush=True)
    print(f"OLMo-7B failed to load with BnB on T4 — Yi-6B is the third GQA family", flush=True)

    data = load_data(300)

    results = {
        "family": "yi",
        "arch": ARCH,
        "base_id": BASE_ID,
        "inst_id": INST_ID,
        "note": "OLMo-7B incompatible with BnB 4-bit on T4 — Yi-6B used as third GQA family",
        "status": "in_progress",
    }

    # ── Base model ──────────────────────────────────────────────────────────
    base_auroc, base_dir = None, None
    try:
        mdl, tok = load_model(BASE_ID)
        n_layers  = mdl.config.num_hidden_layers
        layer_idx = max(0, n_layers - 2)
        d, mu_p, mu_w, auroc = calibrate(mdl, tok, data, layer_idx, BASE_ID)
        base_auroc = auroc
        base_dir   = d
        results.update({
            "base_auroc": round(auroc, 4), "base_mu_p": round(mu_p, 3),
            "base_mu_w": round(mu_w, 3), "probe_layer": layer_idx
        })
        print(f"\n  → BASE AUROC: {auroc:.4f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] base: {e}", flush=True)
        import traceback; traceback.print_exc()
        results["base_error"] = str(e)
    finally:
        try:
            release_model(mdl)
        except Exception:
            pass
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    OUT_FILE.write_text(json.dumps(results, indent=2))

    # ── Instruct model ──────────────────────────────────────────────────────
    inst_auroc, inst_dir = None, None
    try:
        mdl, tok  = load_model(INST_ID)
        n_layers  = mdl.config.num_hidden_layers
        layer_idx = results.get("probe_layer", max(0, n_layers - 2))
        d, mu_p, mu_w, auroc = calibrate(mdl, tok, data, layer_idx, INST_ID)
        inst_auroc = auroc
        inst_dir   = d
        results.update({
            "inst_auroc": round(auroc, 4), "inst_mu_p": round(mu_p, 3),
            "inst_mu_w": round(mu_w, 3)
        })
        print(f"\n  → INST AUROC: {auroc:.4f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] instruct: {e}", flush=True)
        import traceback; traceback.print_exc()
        results["inst_error"] = str(e)
    finally:
        try:
            release_model(mdl)
        except Exception:
            pass

    # ── Verdict ─────────────────────────────────────────────────────────────
    if base_auroc is not None and inst_auroc is not None:
        delta = inst_auroc - base_auroc
        cosim = float(np.dot(base_dir, inst_dir))
        results["delta"]      = round(delta, 4)
        results["cosim_dirs"] = round(cosim, 4)

        if cosim > 0.5:
            cosim_interp = "SAME_DIRECTION (attenuation)"
        elif cosim < 0.1:
            cosim_interp = "NEAR_ORTHOGONAL (rotation — same as MQA)"
        else:
            cosim_interp = "PARTIAL_ROTATION"

        if abs(delta - (-0.036)) < 0.020:
            verdict = "GQA_UNIVERSAL_CONFIRMED"
        elif delta < -0.060:
            verdict = "STRONGER_ATTENUATION"
        elif delta > -0.010:
            verdict = "MINIMAL_ATTENUATION"
        else:
            verdict = "MODERATE_ATTENUATION"

        results["verdict"]      = verdict
        results["cosim_interp"] = cosim_interp

        print(f"\n{'='*60}", flush=True)
        print(f"  base_auroc  = {base_auroc:.4f}", flush=True)
        print(f"  inst_auroc  = {inst_auroc:.4f}", flush=True)
        print(f"  Δ           = {delta:+.4f}", flush=True)
        print(f"  cosim_dirs  = {cosim:.4f}  ({cosim_interp})", flush=True)
        print(f"  VERDICT:      {verdict}", flush=True)
        print(f"{'='*60}", flush=True)
    else:
        results["verdict"] = "INCOMPLETE"

    results["elapsed_s"] = round(time.time() - t0)
    results["status"]    = "complete"
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n[Final] {OUT_FILE}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

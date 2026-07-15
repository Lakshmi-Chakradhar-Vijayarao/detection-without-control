#!/usr/bin/env python3
"""
math_auroc_llama_v1.py — LAW 4 LLAMA REPLICATION
==================================================
SCIENTIFIC QUESTION (Q5, Tier 2 — Laws):
  Does Law 4 (step-1 AUROC ≥ 0.85 on mathematical reasoning) hold on
  the Llama backbone (DeepSeek-R1-Distill-Llama-8B)?

BACKGROUND:
  C039 established AUROC=0.8558 on DeepSeek-R1-Distill-Qwen-1.5B (math_auroc_v2, N=100/class).
  Law 4 claims: "approach-commitment geometry at problem onset predicts final answer correctness
  with AUROC ≥ 0.85 on mathematical reasoning tasks."
  This is currently single-architecture (Qwen distill). Llama replication determines whether
  it generalizes across reasoning model families.

DESIGN:
  Model:   deepseek-ai/DeepSeek-R1-Distill-Llama-8B (32 layers, d=4096, 4-bit on T4)
  Dataset: HuggingFaceH4/MATH-500 (same as C039)
  Labels:  CORRECT (boxed answer matches gold) vs WRONG
  N:       100/class (200 total)
  Layers:  L29 and L30 (proportional to L25/L26 in 28-layer Qwen: round(25/28*32)=29, round(26/28*32)=30)
  Probe:   Fisher+PCA64 (PCA n_components=64, LDA solver=lsqr, shrinkage=auto)

  Step-1 extraction: two-pass (prefill → KV cache → one-token forward with output_hidden_states=True)

KILL CRITERION:
  AUROC < 0.70 at both L29 and L30
  → Law 4 is Qwen-distill-specific; retract architecture-general claim

VERDICT CRITERIA (pre-registered):
  LAW4_REPLICATED:  AUROC ≥ 0.85 at best layer (matches C039 bound)
  LAW4_WEAKENED:    0.70 ≤ AUROC < 0.85 (universal floor holds but math premium gone)
  LAW4_FAILED:      AUROC < 0.70 (Law 4 is architecture-conditional)

CLAIM IMPACT:
  C039: will be updated with Llama cross-family evidence
  Law 4: promoted to CONFIRMED (≥0.85) or downgraded to ≥0.70 universal

GPU: T4 (~4-6h — 8B model in 4-bit, 200 items, full reasoning chains + step-1 extraction)
"""

from __future__ import annotations
import subprocess
subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.46.1"], check=False)

import gc, json, os, random, re, time

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as _np
        if isinstance(obj, _np.integer): return int(obj)
        if isinstance(obj, _np.floating): return float(obj)
        if isinstance(obj, _np.bool_): return bool(obj)
        if isinstance(obj, _np.ndarray): return obj.tolist()
        return super().default(obj)
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
SEED            = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID        = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
N_LAYERS        = 32          # DeepSeek-R1-Distill-Llama-8B has 32 layers
LAYER_A         = 29          # proportional to L25/28 in Qwen: round(25/28 * 32) = 29
LAYER_B         = 30          # proportional to L26/28 in Qwen: round(26/28 * 32) = 30
N_TARGET        = 50          # per class; 100 times out on T4 at 175s/item
POOL_SIZE       = 500
MAX_THINK_TOKENS = 1024        # 512 too short (no boxed answers); 1024 ~90s/item, completes easy/medium
PCA_DIM         = 64
TRAIN_FRAC      = 0.75
N_BOOTSTRAP     = 1000
SAVE_PATH       = "/kaggle/working/math_auroc_llama_v1_results.json"
INTERIM_PATH    = "/kaggle/working/math_auroc_llama_v1_interim.json"
INTERIM_EVERY   = 10          # more frequent saves (8B model slower)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── HF auth ────────────────────────────────────────────────────────────────────
def _hf_login():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            from huggingface_hub import login as _l
            _l(token=v, add_to_git_credential=False)
            print("HF login: OK", flush=True)
            return
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret("HF_TOKEN")
        if v:
            from huggingface_hub import login as _l
            _l(token=v, add_to_git_credential=False)
            print("HF login: OK (kaggle secrets)", flush=True)
    except Exception:
        pass

_hf_login()


# ── Model loading (4-bit for 8B on T4) ────────────────────────────────────────
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"\nLoading {MODEL_ID} with 4-bit quantization...", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    n = mdl.config.num_hidden_layers
    d = mdl.config.hidden_size
    print(f"  Loaded: {n} layers, d={d}", flush=True)
    if n != N_LAYERS:
        print(f"  WARNING: expected {N_LAYERS} layers, got {n}. Adjusting layer indices.", flush=True)
        global LAYER_A, LAYER_B
        LAYER_A = round(25 / 28 * n)
        LAYER_B = round(26 / 28 * n)
        print(f"  Adjusted: LAYER_A={LAYER_A}, LAYER_B={LAYER_B}", flush=True)
    return mdl, tok


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_math() -> list:
    from datasets import load_dataset
    for src, kwargs in [
        ("HuggingFaceH4/MATH-500",    {"split": "test"}),
        ("lighteval/MATH-Hard",        {"split": "test"}),
        ("EleutherAI/hendrycks_math",  {"name": "algebra", "split": "test"}),
    ]:
        try:
            ds = load_dataset(src, **kwargs)
            print(f"Loaded: {src}", flush=True)
            break
        except Exception as e:
            print(f"  {src} failed: {e}", flush=True)
    else:
        raise RuntimeError("All MATH dataset sources failed.")
    items = []
    for row in ds:
        problem  = row.get("problem", row.get("question", ""))
        solution = row.get("solution", row.get("answer", ""))
        if not problem or not solution:
            continue
        m = re.search(r"\\boxed\{([^}]+)\}", solution)
        gold = m.group(1).strip() if m else solution.strip()[-80:]
        items.append({"problem": problem, "gold": gold})
        if len(items) >= POOL_SIZE:
            break
    random.shuffle(items)
    print(f"Pool: {len(items)} MATH problems", flush=True)
    return items


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def math_correct(pred: str, gold: str) -> bool:
    if not pred:
        return False
    p = pred.lower().strip().replace(" ", "")
    g = gold.lower().strip().replace(" ", "")
    return g == p or g in p or p in g


# ── Prompt ─────────────────────────────────────────────────────────────────────
def make_prompt(tok, problem: str) -> str:
    msgs = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user",   "content": problem},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Generation + step-1 HS extraction ─────────────────────────────────────────
def generate_answer(model, tok, problem: str) -> str:
    prompt = make_prompt(tok, problem)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=MAX_THINK_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_step1_hs(model, tok, problem: str, layers: list) -> dict | None:
    """Two-pass: prefill → KV cache → step-1 with output_hidden_states."""
    prompt = make_prompt(tok, problem)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    try:
        with torch.no_grad():
            prefill = model(
                **ids,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            next_tok = prefill.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            gen = model(
                input_ids=next_tok,
                past_key_values=prefill.past_key_values,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states: (n_layers+1,) where index k = layer k-1 output (index 0 = embedding)
        result = {}
        for layer_idx in layers:
            hs_idx = layer_idx + 1  # shift: hidden_states[layer_idx+1] = output of layer layer_idx
            if hs_idx < len(gen.hidden_states):
                result[layer_idx] = gen.hidden_states[hs_idx][0, 0, :].detach().float().cpu().numpy()
        del prefill, gen
        return result if result else None
    except Exception as e:
        print(f"    WARN extract_step1_hs: {e}", flush=True)
        return None


# ── Probe utilities ────────────────────────────────────────────────────────────
def fit_probe(X_tr, y_tr, X_te, y_te, seed=SEED):
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    X_tr_r = pca.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca.transform(X_te.astype(np.float32))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_r, y_tr)
    scores   = lda.decision_function(X_te_r)
    auroc    = float(roc_auc_score(y_te, scores))
    y_shuf   = y_te.copy()
    np.random.RandomState(seed).shuffle(y_shuf)
    shuffled = float(roc_auc_score(y_shuf, scores))
    return round(auroc, 4), round(shuffled, 4), pca, lda


def bootstrap_ci(X, y, pca, lda, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    X_r = pca.transform(X.astype(np.float32))
    scores = lda.decision_function(X_r)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        try:
            aurocs.append(float(roc_auc_score(y[idx], scores[idx])))
        except Exception:
            pass
    if not aurocs:
        return 0.0, 0.0
    return round(sorted(aurocs)[int(0.025*len(aurocs))], 4), round(sorted(aurocs)[int(0.975*len(aurocs))], 4)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_math()
    model, tok = load_model()

    correct_hs = {LAYER_A: [], LAYER_B: []}
    wrong_hs   = {LAYER_A: [], LAYER_B: []}
    n_scanned = n_correct = n_wrong = n_gen_errors = 0

    print(f"\n=== Phase 1: Collection (target {N_TARGET}/class) ===", flush=True)

    for item in pool:
        if n_correct >= N_TARGET and n_wrong >= N_TARGET:
            break

        n_scanned += 1
        problem = item["problem"]
        gold    = item["gold"]

        try:
            gen_text = generate_answer(model, tok, problem)
        except Exception as e:
            n_gen_errors += 1
            print(f"  GEN_ERROR[{n_scanned}]: {e}", flush=True)
            continue

        pred_boxed = extract_boxed(gen_text)
        if not pred_boxed:  # model timed out before producing \boxed{} — skip, don't count as wrong
            continue

        is_correct = math_correct(pred_boxed, gold)

        need_correct = n_correct < N_TARGET
        need_wrong   = n_wrong   < N_TARGET
        if not ((is_correct and need_correct) or (not is_correct and need_wrong)):
            continue

        hs = extract_step1_hs(model, tok, problem, [LAYER_A, LAYER_B])
        if hs is None:
            continue

        if is_correct:
            correct_hs[LAYER_A].append(hs.get(LAYER_A))
            correct_hs[LAYER_B].append(hs.get(LAYER_B))
            n_correct += 1
        else:
            wrong_hs[LAYER_A].append(hs.get(LAYER_A))
            wrong_hs[LAYER_B].append(hs.get(LAYER_B))
            n_wrong += 1

        if (n_correct + n_wrong) % INTERIM_EVERY == 0:
            elapsed = int(time.time() - t0)
            print(f"  [scanned={n_scanned}] CORRECT={n_correct} WRONG={n_wrong}  elapsed={elapsed}s", flush=True)
            with open(INTERIM_PATH, "w") as f:
                json.dump({"n_scanned": n_scanned, "n_correct": n_correct, "n_wrong": n_wrong,
                           "gen_errors": n_gen_errors, "elapsed_s": elapsed}, f, cls=_NpEncoder)

        # Clear GPU cache periodically
        if n_scanned % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed_phase1 = int(time.time() - t0)
    print(f"\nPhase 1 complete: CORRECT={n_correct} WRONG={n_wrong} scanned={n_scanned}", flush=True)
    print(f"  gen_errors={n_gen_errors}  elapsed={elapsed_phase1}s", flush=True)

    if n_correct < 20 or n_wrong < 20:
        raise RuntimeError(f"Insufficient data: CORRECT={n_correct} WRONG={n_wrong}")

    results = {
        "experiment": "MATH_AUROC_LLAMA_V1",
        "model": MODEL_ID,
        "n_layers": N_LAYERS,
        "layer_a": LAYER_A,
        "layer_b": LAYER_B,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_scanned": n_scanned,
        "gen_errors": n_gen_errors,
        "elapsed_phase1_s": elapsed_phase1,
        "layers": {},
    }

    print(f"\n=== Phase 2: Probe fit and AUROC ===", flush=True)

    for layer_idx, layer_name in [(LAYER_A, f"L{LAYER_A}"), (LAYER_B, f"L{LAYER_B}")]:
        Xc_list = [x for x in correct_hs[layer_idx] if x is not None]
        Xw_list = [x for x in wrong_hs[layer_idx]   if x is not None]
        if not Xc_list or not Xw_list:
            print(f"  {layer_name}: no hidden states collected. Skipping.", flush=True)
            continue

        Xc = np.stack(Xc_list).astype(np.float32)
        Xw = np.stack(Xw_list).astype(np.float32)
        n_use = min(len(Xc), len(Xw))
        Xc, Xw = Xc[:n_use], Xw[:n_use]
        X = np.concatenate([Xc, Xw])
        y = np.array([1]*n_use + [0]*n_use)

        rng_split = np.random.RandomState(SEED)
        idx = np.arange(len(y)); rng_split.shuffle(idx)
        n_train = int(TRAIN_FRAC * len(idx))
        tr_idx, te_idx = idx[:n_train], idx[n_train:]

        auroc, shuffled, pca, lda = fit_probe(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        ci_lo, ci_hi = bootstrap_ci(X[te_idx], y[te_idx], pca, lda)

        # Verdict for this layer
        kill = auroc < 0.70 or shuffled >= 0.60
        if auroc >= 0.85:
            layer_verdict = "LAW4_REPLICATED"
        elif auroc >= 0.70:
            layer_verdict = "LAW4_WEAKENED"
        else:
            layer_verdict = "LAW4_FAILED"

        print(f"  {layer_name}: AUROC={auroc:.4f}  CI=[{ci_lo},{ci_hi}]  "
              f"shuffled={shuffled:.4f}  verdict={layer_verdict}", flush=True)

        results["layers"][layer_name] = {
            "layer_idx": layer_idx,
            "n_per_class": n_use,
            "n_train": n_train,
            "n_test": len(te_idx),
            "auroc": auroc,
            "ci_95": [ci_lo, ci_hi],
            "shuffled": shuffled,
            "kill_triggered": kill,
            "layer_verdict": layer_verdict,
        }

    total_elapsed = int(time.time() - t0)
    results["elapsed_total_s"] = total_elapsed
    results["elapsed_total_min"] = round(total_elapsed / 60, 1)

    # Overall verdict
    layer_aurocs = {k: v["auroc"] for k, v in results["layers"].items()}
    if not layer_aurocs:
        results["overall_verdict"] = "NO_DATA"
    else:
        best_auroc = max(layer_aurocs.values())
        best_layer = max(layer_aurocs, key=lambda k: layer_aurocs[k])
        results["best_auroc"] = best_auroc
        results["best_layer"] = best_layer
        if best_auroc >= 0.85:
            results["overall_verdict"] = "LAW4_REPLICATED"
        elif best_auroc >= 0.70:
            results["overall_verdict"] = "LAW4_WEAKENED"
        else:
            results["overall_verdict"] = "LAW4_FAILED"
        results["c039_comparison"] = {
            "c039_auroc": 0.8558,
            "c039_model": "DeepSeek-R1-Distill-Qwen-1.5B",
            "c039_n_per_class": 100,
            "llama_best_auroc": best_auroc,
            "delta": round(best_auroc - 0.8558, 4),
        }

        print(f"\n{'='*60}", flush=True)
        print(f"BEST AUROC: {best_auroc:.4f} at {best_layer}", flush=True)
        print(f"C039 comparison: {best_auroc:.4f} vs 0.8558 (Qwen, N=100/class)", flush=True)
        print(f"OVERALL VERDICT: {results['overall_verdict']}", flush=True)
        print(f"Elapsed: {results['elapsed_total_min']} min", flush=True)
        print(f"{'='*60}", flush=True)

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    print(f"\nResults saved to {SAVE_PATH}", flush=True)


if __name__ == "__main__":
    main()

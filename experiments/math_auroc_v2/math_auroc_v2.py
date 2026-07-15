#!/usr/bin/env python3
"""
math_auroc_v2.py — EXP_MATH_AUROC_V2

SCIENTIFIC QUESTION:
  C039 established AUROC=0.9111 at step-1 (first generated token, before any reasoning)
  for CORRECT vs WRONG final answers on MATH competition problems using
  DeepSeek-R1-Distill-Qwen-1.5B (N=30/class, small-N). This replication uses N=100/class
  for proper statistical power and 95% bootstrap CI.

KILL CRITERION:
  AUROC < 0.70 OR shuffled control >= 0.60
  → C039 finding is a small-N artifact; approach-commitment geometry at problem onset
  does not reliably predict final answer correctness.

DESIGN:
  Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (same as C039)
  Dataset: HuggingFaceH4/MATH-500 (500 competition math problems)
  Label: CORRECT (final boxed answer matches gold) vs WRONG (doesn't match)
  N: 100/class (200 total)
  Layers: L25 and L26 (C039 optimal: both gave 0.9111)
  Probe: Fisher+PCA64 (PCA n_components=64, LDA solver=lsqr, shrinkage=auto)

  Protocol:
  Phase 1 (Collection):
    For each MATH problem in shuffled order:
      - Generate full reasoning chain (up to MAX_THINK_TOKENS tokens)
      - Extract final \boxed{...} answer
      - Check correctness vs gold answer
      - If CC/CW slots still open: extract step-1 HS at L25 and L26 via two-pass forward
      - Stop when 100 CORRECT + 100 WRONG collected
    Intermediate save every 20 items.

  Phase 2 (Probe fit and AUROC):
    - Split 75/25 (train/test), stratified
    - Fit PCA64 then LDA at L25, evaluate on test → AUROC, shuffled AUROC
    - Repeat at L26
    - Bootstrap CI: 1000 samples, 95% percentile interval

  Step-1 definition:
    Prefill the full prompt (system + problem), then run ONE forward pass on the
    first generated token with KV cache active. The hidden state at L25/L26 at this
    step is the "step-1 signal" — before any reasoning tokens are generated.

OUTPUT: /kaggle/working/math_auroc_v2_results.json
GPU: T4. Expected ~3-5h.
"""

from __future__ import annotations
import gc, json, os, random, re, time
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID        = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
N_TARGET        = 100           # per class
POOL_SIZE       = 500           # MATH-500 (all problems)
MAX_THINK_TOKENS = 2048
LAYER_A         = 25            # C039 optimal L25
LAYER_B         = 26            # C039 optimal L26
PCA_DIM         = 64
TRAIN_FRAC      = 0.75
N_BOOTSTRAP     = 1000
SAVE_PATH       = "/kaggle/working/math_auroc_v2_results.json"
INTERIM_PATH    = "/kaggle/working/math_auroc_v2_interim.json"
INTERIM_EVERY   = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── HF auth ───────────────────────────────────────────────────────────────────
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


# ── Data ──────────────────────────────────────────────────────────────────────
def load_math() -> list:
    from datasets import load_dataset
    for src, kwargs in [
        ("HuggingFaceH4/MATH-500",    {"split": "test"}),
        ("lighteval/MATH-Hard",        {"split": "test", "streaming": True}),
        ("EleutherAI/hendrycks_math",  {"name": "algebra", "split": "test", "streaming": True}),
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
    """Extract last \\boxed{...} content from generated text."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def math_correct(pred_boxed: str, gold: str) -> bool:
    """Fuzzy match: gold contained in prediction or exact match after normalization."""
    if not pred_boxed:
        return False
    p = pred_boxed.lower().strip().replace(" ", "")
    g = gold.lower().strip().replace(" ", "")
    return g == p or g in p or p in g


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {MODEL_ID} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    n = mdl.config.num_hidden_layers
    d = mdl.config.hidden_size
    print(f"  Loaded: {n} layers, d={d}", flush=True)
    return mdl, tok


def make_prompt(tok, problem: str) -> str:
    msgs = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user",   "content": problem},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_answer(model, tok, problem: str) -> str:
    """Generate full reasoning chain; return decoded text."""
    prompt = make_prompt(tok, problem)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
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
    """
    Two-pass extraction: prefill prompt → step-1 hidden states at specified layers.
    Returns dict {layer_idx: np.ndarray(d,)} or None on error.
    """
    prompt = make_prompt(tok, problem)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    try:
        with torch.no_grad():
            # Pass 1: prefill — get KV cache + first token prediction
            prefill = model(
                **ids,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            next_tok = prefill.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

            # Pass 2: step-1 — first generated token with KV cache
            gen = model(
                input_ids=next_tok,
                past_key_values=prefill.past_key_values,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states: tuple length n_layers+1; index 0=embedding, k=layer k-1 output
        result = {}
        for layer_idx in layers:
            result[layer_idx] = gen.hidden_states[layer_idx + 1][0, 0, :].detach().float().cpu().numpy()
        return result
    except Exception as e:
        print(f"    WARN extract_step1_hs: {e}", flush=True)
        return None


# ── Probe utilities ───────────────────────────────────────────────────────────
def fit_probe(X_tr, y_tr, X_te, y_te, seed=SEED):
    n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    X_tr_r = pca.fit_transform(X_tr.astype(np.float32))
    X_te_r = pca.transform(X_te.astype(np.float32))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_r, y_tr)
    scores   = lda.decision_function(X_te_r)
    auroc    = float(roc_auc_score(y_te, scores))
    y_shuf   = y_te.copy(); rng = np.random.RandomState(seed); rng.shuffle(y_shuf)
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
        return (0.0, 0.0)
    aurocs = sorted(aurocs)
    lo = aurocs[int(0.025 * len(aurocs))]
    hi = aurocs[int(0.975 * len(aurocs))]
    return round(lo, 4), round(hi, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_math()
    model, tok = load_model()

    correct_hs = {LAYER_A: [], LAYER_B: []}
    wrong_hs   = {LAYER_A: [], LAYER_B: []}
    n_scanned = 0
    n_correct = 0
    n_wrong   = 0
    n_gen_errors = 0

    print(f"\n=== Phase 1: Collection (target {N_TARGET}/class) ===", flush=True)

    for item in pool:
        if n_correct >= N_TARGET and n_wrong >= N_TARGET:
            break

        n_scanned += 1
        problem = item["problem"]
        gold    = item["gold"]

        # Generate full answer to get label
        try:
            gen_text = generate_answer(model, tok, problem)
        except Exception as e:
            n_gen_errors += 1
            print(f"  GEN_ERROR[{n_scanned}]: {e}", flush=True)
            continue

        pred_boxed = extract_boxed(gen_text)
        is_correct = math_correct(pred_boxed, gold)

        # Decide if we still need this class
        need_correct = n_correct < N_TARGET
        need_wrong   = n_wrong < N_TARGET
        use_item = (is_correct and need_correct) or (not is_correct and need_wrong)

        if not use_item:
            continue

        # Extract step-1 HS at both layers
        hs = extract_step1_hs(model, tok, problem, [LAYER_A, LAYER_B])
        if hs is None:
            continue

        if is_correct:
            correct_hs[LAYER_A].append(hs[LAYER_A])
            correct_hs[LAYER_B].append(hs[LAYER_B])
            n_correct += 1
        else:
            wrong_hs[LAYER_A].append(hs[LAYER_A])
            wrong_hs[LAYER_B].append(hs[LAYER_B])
            n_wrong += 1

        if (n_correct + n_wrong) % INTERIM_EVERY == 0:
            elapsed = int(time.time() - t0)
            print(f"  [scanned={n_scanned}] CORRECT={n_correct} WRONG={n_wrong}  elapsed={elapsed}s", flush=True)
            # Interim save
            interim = {
                "n_scanned": n_scanned, "n_correct": n_correct, "n_wrong": n_wrong,
                "gen_errors": n_gen_errors, "elapsed_s": elapsed,
            }
            with open(INTERIM_PATH, "w") as f:
                json.dump(interim, f)

    elapsed_phase1 = int(time.time() - t0)
    print(f"\nPhase 1 complete: CORRECT={n_correct} WRONG={n_wrong} scanned={n_scanned}", flush=True)
    print(f"  gen_errors={n_gen_errors}  elapsed={elapsed_phase1}s", flush=True)

    if n_correct < 20 or n_wrong < 20:
        raise RuntimeError(f"Insufficient data: CORRECT={n_correct} WRONG={n_wrong}")

    results = {
        "n_correct": n_correct, "n_wrong": n_wrong, "n_scanned": n_scanned,
        "gen_errors": n_gen_errors, "elapsed_phase1_s": elapsed_phase1,
        "layers": {},
    }

    print(f"\n=== Phase 2: Probe fit and AUROC ===", flush=True)

    for layer_idx, layer_name in [(LAYER_A, f"L{LAYER_A}"), (LAYER_B, f"L{LAYER_B}")]:
        Xc = np.stack(correct_hs[layer_idx]).astype(np.float32)  # (n_correct, d)
        Xw = np.stack(wrong_hs[layer_idx]).astype(np.float32)    # (n_wrong, d)
        n_use = min(len(Xc), len(Xw))
        Xc, Xw = Xc[:n_use], Xw[:n_use]
        X = np.concatenate([Xc, Xw], axis=0)
        y = np.array([1] * n_use + [0] * n_use)

        # Stratified 75/25 split
        idx = np.arange(len(y))
        rng_split = np.random.RandomState(SEED)
        rng_split.shuffle(idx)
        n_train = int(TRAIN_FRAC * len(idx))
        tr_idx, te_idx = idx[:n_train], idx[n_train:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        auroc, shuffled, pca, lda = fit_probe(X_tr, y_tr, X_te, y_te)
        ci_lo, ci_hi = bootstrap_ci(X_te, y_te, pca, lda)

        kill_triggered = auroc < 0.70 or shuffled >= 0.60
        print(f"  {layer_name}: AUROC={auroc:.4f}  CI=[{ci_lo},{ci_hi}]  "
              f"shuffled={shuffled:.4f}  N_test={len(y_te)}  kill={kill_triggered}", flush=True)

        results["layers"][layer_name] = {
            "layer_idx": layer_idx,
            "n_per_class": n_use,
            "n_train": n_train,
            "n_test": len(y_te),
            "auroc": auroc,
            "ci_95": [ci_lo, ci_hi],
            "shuffled": shuffled,
            "kill_triggered": kill_triggered,
        }

    total_elapsed = int(time.time() - t0)
    results["elapsed_total_s"] = total_elapsed
    results["elapsed_total_min"] = round(total_elapsed / 60, 1)

    # Summary
    la_res = results["layers"][f"L{LAYER_A}"]
    lb_res = results["layers"][f"L{LAYER_B}"]
    best_auroc = max(la_res["auroc"], lb_res["auroc"])
    best_layer = f"L{LAYER_A}" if la_res["auroc"] >= lb_res["auroc"] else f"L{LAYER_B}"
    kill = la_res["kill_triggered"] or lb_res["kill_triggered"]

    print(f"\n{'='*60}", flush=True)
    print(f"BEST AUROC: {best_auroc:.4f} at {best_layer}", flush=True)
    print(f"kill_triggered: {kill}", flush=True)
    print(f"C039 comparison: {best_auroc:.4f} vs 0.9111 (N=30/class)", flush=True)
    print(f"Elapsed: {results['elapsed_total_min']} min", flush=True)
    print(f"{'='*60}", flush=True)

    results["best_auroc"] = best_auroc
    results["best_layer"] = best_layer
    results["kill_triggered"] = kill
    results["c039_comparison"] = {"c039_auroc": 0.9111, "c039_n_per_class": 30}

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {SAVE_PATH}", flush=True)


if __name__ == "__main__":
    main()

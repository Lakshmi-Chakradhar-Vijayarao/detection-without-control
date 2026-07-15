#!/usr/bin/env python3
"""
exp_b_fullspace_fisher.py — FULL-SPACE FISHER LDA ABLATION
===========================================================
SCIENTIFIC QUESTION (Q10, Tier 3 — Mechanism):
  Is PCA64 compression a bottleneck on measurable O, or is the signal genuinely linear
  in the full representation space?

BACKGROUND:
  C002 (CONFIRMED): No nonlinear probe recovery over Fisher+PCA64.
  BUT: PCA64 was applied before probing. PCA discards ~96% of variance in 1536-d Qwen space.
  C002 proves "no nonlinear recovery within 64 PCA dimensions" — not "no recovery in full space."

  This experiment tests all probe variants on the same extraction:
    1. Fisher+PCA64 (standard — reference, replicates C001)
    2. Fisher+PCA128, Fisher+PCA256, Fisher+PCA512
    3. Fisher (no PCA, full 1536-d, LSQR shrinkage='auto')
    4. LogReg (full space, linear)
    5. MLP (2 layers, nonlinear, full space)

  Also measures:
    - Covariance condition number at each PCA dim (explains why PCA helps conditioning)
    - Cosine similarity between PCA64 discriminant direction and full-space discriminant direction
      (do they read the same direction, or different parts of the space?)

VERDICT CRITERIA (pre-registered):
  PCA_BOTTLENECK:  full-space Fisher AUROC > PCA64 AUROC by > 0.05
  PCA_VALIDATED:   full-space Fisher AUROC within 0.02 of PCA64 AUROC
  NONLINEAR_RECOVERY: MLP AUROC > Fisher+PCA64 AUROC by > 0.05 in full space
  LINEAR_SIGNAL:   no nonlinear recovery in full space (MLP ≈ Fisher+PCA64)

CLAIM IMPACT:
  Updates C002 (NO_RECOVERY) — expands or constrains scope to "within/beyond PCA64 projection"

MODEL: Qwen/Qwen2.5-1.5B-Instruct (same as C001-C004 primary experiments)
TASK:  TriviaQA bilateral oracle, N=150/class, pool=5000
LAYER: 26, Step 1

GPU: T4 (~45-60 min — one model, ~300 items)
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("GPU required. Exiting.")

MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX   = 26
N_TARGET    = 175
TRAIN_FRAC  = 0.75
POOL_SIZE   = 8000
N_BOOTSTRAP = 500
N_SHUFFLED  = 3
MAX_GEN     = 60
MAX_CTX     = 800
PARAM_MIN_F1 = 0.50
CTX_MAX_F1   = 0.05
CTX_MIN_CTX  = 0.50
OUTPUT_FILE = "/kaggle/working/exp_b_fullspace_fisher_results.json"

PCA_DIMS    = [64, 128, 256, 512]  # ablation over compression levels


# ── HF token ─────────────────────────────────────────────────────────────────
def _get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_triviaqa():
    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        q = ex["question"]
        answers = ex["answer"]["aliases"] if ex["answer"]["aliases"] else [ex["answer"]["value"]]
        passage = ""
        if ex.get("entity_pages", {}).get("wiki_context"):
            passage = ex["entity_pages"]["wiki_context"][0]
        items.append({"question": q, "answers": answers, "passage": passage})
    random.shuffle(items)
    print(f"  Loaded {len(items)} items")
    return items


def token_f1(pred: str, gold_list: list[str]) -> float:
    pred_tokens = set(pred.lower().split())
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_list:
        gold_tokens = set(gold.lower().split())
        if not gold_tokens:
            continue
        common = len(pred_tokens & gold_tokens)
        if common == 0:
            continue
        p, r = common / len(pred_tokens), common / len(gold_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(hf_token):
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


# ── Hidden state extraction ───────────────────────────────────────────────────
@torch.no_grad()
def extract_hidden_state(model, tokenizer, prompt: str) -> np.ndarray | None:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[1] > 1800:
        return None

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_GEN,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    if not (out.hidden_states and len(out.hidden_states) > 1):
        return None
    step1_layers = out.hidden_states[1]
    if LAYER_IDX >= len(step1_layers):
        return None
    hs = step1_layers[LAYER_IDX][0, -1, :].float().cpu().numpy()
    return hs


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[1] > 1800:
        return ""
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_GEN,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ── Bilateral oracle + hidden state collection ────────────────────────────────
def collect_hidden_states(model, tokenizer, items, n_target=N_TARGET):
    print(f"\nCollecting hidden states (target {n_target}/class)...")
    param_hs, ctxdep_hs = [], []
    scanned = 0
    t0 = time.time()

    for item in items:
        if len(param_hs) >= n_target and len(ctxdep_hs) >= n_target:
            break
        if scanned >= POOL_SIZE:
            break
        scanned += 1

        q, answers, passage = item["question"], item["answers"], item.get("passage", "")

        # Format no-context prompt
        msgs_nc = [{"role": "user", "content": f"Answer in a few words: {q}"}]
        prompt_nc = tokenizer.apply_chat_template(msgs_nc, tokenize=False, add_generation_prompt=True)
        text_nc = generate_text(model, tokenizer, prompt_nc)
        f1_nc = token_f1(text_nc, answers)

        if f1_nc >= PARAM_MIN_F1 and len(param_hs) < n_target:
            hs = extract_hidden_state(model, tokenizer, prompt_nc)
            if hs is not None:
                param_hs.append(hs)
            continue

        if f1_nc <= CTX_MAX_F1 and passage:
            ctx = passage[:MAX_CTX]
            msgs_wc = [{"role": "user", "content": f"Context: {ctx}\nAnswer: {q}"}]
            prompt_wc = tokenizer.apply_chat_template(msgs_wc, tokenize=False, add_generation_prompt=True)
            text_wc = generate_text(model, tokenizer, prompt_wc)
            f1_wc = token_f1(text_wc, answers)
            if f1_wc >= CTX_MIN_CTX and len(ctxdep_hs) < n_target:
                hs = extract_hidden_state(model, tokenizer, prompt_nc)
                if hs is not None:
                    ctxdep_hs.append(hs)

        if scanned % 200 == 0:
            print(f"  scanned={scanned}  PARAM={len(param_hs)}  CTX_DEP={len(ctxdep_hs)}  ({time.time()-t0:.0f}s)")

    print(f"Done. PARAM={len(param_hs)}, CTX_DEP={len(ctxdep_hs)}")
    return np.array(param_hs), np.array(ctxdep_hs)


# ── Probe variants ────────────────────────────────────────────────────────────
def split_data(X_pos, X_neg, seed=SEED):
    n = min(len(X_pos), len(X_neg))
    rng = np.random.default_rng(seed)
    idx_p = rng.permutation(len(X_pos))[:n]
    idx_n = rng.permutation(len(X_neg))[:n]
    Xp, Xn = X_pos[idx_p], X_neg[idx_n]
    n_train = int(n * TRAIN_FRAC)
    X_train = np.vstack([Xp[:n_train], Xn[:n_train]])
    y_train = np.array([1] * n_train + [0] * n_train)
    X_test  = np.vstack([Xp[n_train:], Xn[n_train:]])
    y_test  = np.array([1] * (n - n_train) + [0] * (n - n_train))
    return X_train, y_train, X_test, y_test


def fisher_pca(X_train, y_train, X_test, n_components):
    n_components_actual = min(n_components, X_train.shape[0] - 1, X_train.shape[1] - 1)
    if n_components_actual < n_components:
        print(f"    (capped n_components {n_components} → {n_components_actual} due to n_train={X_train.shape[0]})")
    pca = PCA(n_components=n_components_actual, random_state=SEED)
    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(Xtr_p, y_train)
    scores = lda.decision_function(Xte_p)
    # Also return discriminant direction in PCA space (then project back to original space)
    direction_pca = lda.coef_[0]  # [n_components]
    direction_orig = pca.components_.T @ direction_pca  # [hidden_dim]
    direction_orig /= (np.linalg.norm(direction_orig) + 1e-10)
    return scores, direction_orig, pca


def fisher_fullspace(X_train, y_train, X_test):
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train, y_train)
    scores = lda.decision_function(X_test)
    direction = lda.coef_[0]
    direction /= (np.linalg.norm(direction) + 1e-10)
    return scores, direction


def logreg_fullspace(X_train, y_train, X_test):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    scores = clf.decision_function(X_test)
    direction = clf.coef_[0]
    direction /= (np.linalg.norm(direction) + 1e-10)
    return scores, direction


def mlp_fullspace(X_train, y_train, X_test):
    hidden = 256
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden, 64),
        activation="relu",
        max_iter=200,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.15,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    return scores


def condition_number(X, n_components):
    """Estimated condition number of covariance matrix after PCA compression."""
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(X)
    explained_var = pca.explained_variance_
    return float(explained_var[0] / (explained_var[-1] + 1e-30))


def bootstrap_auroc_from_scores(y_test, scores, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    n = len(y_test)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            a = roc_auc_score(y_test[idx], scores[idx])
            boot.append(a)
        except ValueError:
            pass
    if len(boot) < 10:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(boot)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def shuffled_control_auroc(X_pos, X_neg, probe_fn, n_seeds=N_SHUFFLED):
    rng = np.random.default_rng(SEED + 100)
    aurocs = []
    for _ in range(n_seeds):
        X_train, y_train, X_test, y_test = split_data(X_pos, X_neg, seed=int(rng.integers(0, 2**31)))
        y_shuf = rng.permutation(y_train)
        try:
            scores = probe_fn(X_train, y_shuf, X_test)
            aurocs.append(roc_auc_score(y_test, scores))
        except Exception:
            pass
    return float(np.mean(aurocs)) if aurocs else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("EXP_B_FULLSPACE_FISHER — Full-Space Fisher LDA Ablation")
    print("=" * 70)
    print(f"Model: {MODEL_ID}  |  Layer: {LAYER_IDX}  |  N_target: {N_TARGET}/class")

    hf_token = _get_hf_token()
    items = load_triviaqa()
    model, tokenizer = load_model(hf_token)

    # Collect hidden states
    param_hs, ctxdep_hs = collect_hidden_states(model, tokenizer, items)

    # Free model memory
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(param_hs) < 40 or len(ctxdep_hs) < 40:
        print(f"⚠ Insufficient samples: PARAM={len(param_hs)}, CTX_DEP={len(ctxdep_hs)}")
        return

    print(f"\nRunning probe ablation on {len(param_hs)} PARAM + {len(ctxdep_hs)} CTX_DEP samples...")
    print(f"Hidden dimension: {param_hs.shape[1]}")

    X_train, y_train, X_test, y_test = split_data(param_hs, ctxdep_hs)
    results = {"probes": {}, "direction_cosims": {}}

    # ── PCA variants ──────────────────────────────────────────────────────────
    directions = {}
    for n_comp in PCA_DIMS:
        label = f"Fisher+PCA{n_comp}"
        print(f"\n  Running {label}...")
        try:
            scores, direction, pca = fisher_pca(X_train, y_train, X_test, n_comp)
            auroc = roc_auc_score(y_test, scores)
            auroc_mean, ci_low, ci_high = bootstrap_auroc_from_scores(y_test, scores)
            cond = condition_number(X_train, n_comp)
            explained = float(PCA(n_components=n_comp, random_state=SEED).fit(X_train).explained_variance_ratio_.sum())
            # Shuffled control
            shuf = shuffled_control_auroc(
                param_hs, ctxdep_hs,
                lambda Xtr, ytr, Xte: fisher_pca(Xtr, ytr, Xte, n_comp)[0]
            )
            results["probes"][label] = {
                "auroc": round(float(auroc), 4),
                "bootstrap_mean": round(float(auroc_mean), 4),
                "ci_95_low": round(float(ci_low), 4),
                "ci_95_high": round(float(ci_high), 4),
                "shuffled_auroc": round(float(shuf), 4),
                "condition_number": round(float(cond), 2),
                "explained_variance_ratio": round(float(explained), 4),
                "n_components_requested": n_comp,
                "n_components_actual": int(pca.n_components_),
            }
            directions[label] = direction
            if n_comp == 64:
                results["reference_auroc_pca64"] = round(float(auroc), 4)
            print(f"    AUROC={auroc:.4f}  shuffled={shuf:.4f}  cond={cond:.1f}  var={explained:.3f}")
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            results["probes"][label] = {"error": str(e)}

    # ── Full-space Fisher ─────────────────────────────────────────────────────
    print("\n  Running Fisher (no PCA, full space)...")
    try:
        scores_fs, direction_fs = fisher_fullspace(X_train, y_train, X_test)
        auroc_fs = roc_auc_score(y_test, scores_fs)
        auroc_fs_mean, ci_l_fs, ci_h_fs = bootstrap_auroc_from_scores(y_test, scores_fs)
        shuf_fs = shuffled_control_auroc(
            param_hs, ctxdep_hs,
            lambda Xtr, ytr, Xte: fisher_fullspace(Xtr, ytr, Xte)[0]
        )
        results["probes"]["Fisher_full"] = {
            "auroc": round(float(auroc_fs), 4),
            "bootstrap_mean": round(float(auroc_fs_mean), 4),
            "ci_95_low": round(float(ci_l_fs), 4),
            "ci_95_high": round(float(ci_h_fs), 4),
            "shuffled_auroc": round(float(shuf_fs), 4),
            "note": "LSQR shrinkage='auto', no PCA compression",
        }
        directions["Fisher_full"] = direction_fs
        print(f"    AUROC={auroc_fs:.4f}  shuffled={shuf_fs:.4f}")
    except Exception as e:
        print(f"    ⚠ Error in full-space Fisher: {e}")
        results["probes"]["Fisher_full"] = {"error": str(e)}

    # ── LogReg full-space ─────────────────────────────────────────────────────
    print("\n  Running LogReg (full space)...")
    try:
        scores_lr, direction_lr = logreg_fullspace(X_train, y_train, X_test)
        auroc_lr = roc_auc_score(y_test, scores_lr)
        auroc_lr_mean, ci_l_lr, ci_h_lr = bootstrap_auroc_from_scores(y_test, scores_lr)
        results["probes"]["LogReg_full"] = {
            "auroc": round(float(auroc_lr), 4),
            "bootstrap_mean": round(float(auroc_lr_mean), 4),
            "ci_95_low": round(float(ci_l_lr), 4),
            "ci_95_high": round(float(ci_h_lr), 4),
        }
        directions["LogReg_full"] = direction_lr
        print(f"    AUROC={auroc_lr:.4f}")
    except Exception as e:
        print(f"    ⚠ Error in LogReg: {e}")
        results["probes"]["LogReg_full"] = {"error": str(e)}

    # ── MLP full-space ────────────────────────────────────────────────────────
    print("\n  Running MLP (2 layers, full space)...")
    try:
        scores_mlp = mlp_fullspace(X_train, y_train, X_test)
        auroc_mlp = roc_auc_score(y_test, scores_mlp)
        results["probes"]["MLP_full"] = {
            "auroc": round(float(auroc_mlp), 4),
            "note": "hidden_sizes=(256,64), relu, early_stopping",
        }
        print(f"    AUROC={auroc_mlp:.4f}")
    except Exception as e:
        print(f"    ⚠ Error in MLP: {e}")
        results["probes"]["MLP_full"] = {"error": str(e)}

    # ── Direction cosine similarities ─────────────────────────────────────────
    print("\n  Computing discriminant direction cosine similarities...")
    ref_dir = directions.get("Fisher+PCA64")
    full_dir = directions.get("Fisher_full")
    if ref_dir is not None and full_dir is not None:
        cosim = float(np.dot(ref_dir, full_dir) / (np.linalg.norm(ref_dir) * np.linalg.norm(full_dir) + 1e-10))
        results["direction_cosims"]["PCA64_vs_full"] = round(abs(cosim), 4)
        print(f"    PCA64 vs full-space cosim: {cosim:.4f}")

    for label, direction in directions.items():
        if label != "Fisher+PCA64" and ref_dir is not None:
            cosim = float(np.dot(ref_dir, direction) / (np.linalg.norm(ref_dir) * np.linalg.norm(direction) + 1e-10))
            results["direction_cosims"][f"PCA64_vs_{label}"] = round(abs(cosim), 4)

    # ── Verdict ───────────────────────────────────────────────────────────────
    ref64 = results.get("reference_auroc_pca64", float("nan"))
    full_auroc = results["probes"].get("Fisher_full", {}).get("auroc", float("nan"))
    mlp_auroc = results["probes"].get("MLP_full", {}).get("auroc", float("nan"))

    if not (np.isnan(ref64) or np.isnan(full_auroc)):
        diff = full_auroc - ref64
        if diff > 0.05:
            verdict = "PCA_BOTTLENECK"
        elif abs(diff) <= 0.02:
            verdict = "PCA_VALIDATED"
        else:
            verdict = f"PARTIAL_DIFF_{diff:+.3f}"
    else:
        verdict = "INCOMPLETE"

    nonlinear_recovery = "NONLINEAR_RECOVERY" if (not np.isnan(mlp_auroc) and mlp_auroc - ref64 > 0.05) else "NO_NONLINEAR_RECOVERY"

    results["verdict"] = verdict
    results["nonlinear_verdict"] = nonlinear_recovery
    results["n_param"] = len(param_hs)
    results["n_ctxdep"] = len(ctxdep_hs)
    results["hidden_dim"] = int(param_hs.shape[1])
    results["experiment"] = "EXP_B_FULLSPACE_FISHER"
    results["date"] = time.strftime("%Y-%m-%d")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else x)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for label, pr in results["probes"].items():
        if "auroc" in pr:
            shuf = pr.get("shuffled_auroc", "—")
            shuf_str = f"shuffled={shuf:.4f}" if isinstance(shuf, float) else f"shuffled={shuf}"
            print(f"  {label:<20} AUROC={pr['auroc']:.4f}  {shuf_str}")
    print(f"\n  VERDICT:   {verdict}")
    print(f"  NONLINEAR: {nonlinear_recovery}")
    print(f"  PCA64 vs full cosim: {results['direction_cosims'].get('PCA64_vs_full', 'N/A')}")
    print(f"\n  C002 scope: ", end="")
    if verdict == "PCA_BOTTLENECK":
        print("EXPAND — nonlinear recovery exists in full space; C002 only proves no recovery in PCA64 subspace")
    elif verdict == "PCA_VALIDATED":
        print("MAINTAIN — full-space Fisher matches PCA64; signal is genuinely linear and concentrated in top PCs")
    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

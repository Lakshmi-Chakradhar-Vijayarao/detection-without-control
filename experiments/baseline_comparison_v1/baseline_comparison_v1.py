#!/usr/bin/env python3
"""
baseline_comparison_v1.py — EXP_1_BASELINE_COMPARISON_V1
EIG rank #2. Head-to-head: Fisher+PCA64 vs three standard baselines
on the L2 false-certainty task (CC vs CW, entropy-matched).

SCIENTIFIC QUESTION:
  Is Fisher+PCA64 actually better than simpler alternatives, or is the L2
  result an artefact of cherry-picking a complex probe?

BASELINES:
  B1 — Verbalized uncertainty: ask model "How confident are you? (0-100)"
       Score = 100 - verbalized_confidence
  B2 — Self-consistency (N=5 samples): fraction of 5 samples that match
       the greedy answer.  Score = 1 - consistency (higher = less consistent)
  B3 — Top-1 probability: -log P(first_token | prompt)

SETUP:
  Model: Qwen2.5-1.5B-Instruct
  Layer: 26
  N=100/class (CC/CW, entropy-matched, same protocol as false_certainty_v2)
  Pool: 8000
  Bootstrap CI (n=1000) for Fisher only
  Metric: AUROC (label=1 for CC, label=0 for CW)

DECISION TREE:
  Fisher > all 3 baselines by > 0.05:
    → Fisher captures something inaccessible to behavioral signals → C017 strengthened
  Fisher ≈ B3 (top-1 prob):
    → Fisher redundant; simplest baseline sufficient → scope contraction
  Fisher > B3 but < 0.05:
    → Fisher marginal; note in paper as caveat

GPU: T4 (~4h)
"""

from __future__ import annotations
import gc, json, os, random, time

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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX       = 26
N_CC_TARGET     = 100
N_CW_TARGET     = 100
POOL_SIZE       = 8_000
N_SC_SAMPLES    = 5
N_BOOTSTRAP     = 1000
TRAIN_FRAC      = 0.75
PCA_DIM         = 64
MAX_NEW         = 60
PARAM_MIN_F1    = 0.50
CW_MAX_F1       = 0.05
ENT_HALF        = 0.30

SAVE_PATH       = "/kaggle/working/baseline_comparison_v1_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


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
def _get_hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t and t.startswith("hf_"):
        return t
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t and t.startswith("hf_"):
            return t
    except Exception:
        pass
    return None

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _hf_tok = _get_hf_token()
    if _hf_tok:
        from huggingface_hub import login
        login(token=_hf_tok, add_to_git_credential=False)
        print("HF login OK", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=_hf_tok, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, token=_hf_tok, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def prompt_nc(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_entropy_and_logprob(model, tokenizer, prompt: str):
    """Single prefill forward pass — returns (entropy, top1_logprob) for entropy matching.
    Uses next-token distribution before generation (correct for output-entropy baselines)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1, :].float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
    probs  = torch.softmax(logits, dim=-1)
    ent    = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
    if not np.isfinite(ent):
        ent = 0.0
    top1_logprob = float(torch.log(probs.max() + 1e-10).item())
    return ent, top1_logprob


def generate_with_step1_hs(model, tokenizer, prompt: str, layer_idx: int,
                            do_sample: bool = False, temperature: float = 1.0):
    """Generate text AND capture hidden state at step-1 (first new token).
    With KV cache each decode step processes one token (shape[1]==1 in the hook);
    the first such call is step-1, consistent with C001 (AUROC=0.854)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if hs_out[0] is None and x.shape[1] == 1:  # step-1 only
            hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()

    h = model.model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW,
            do_sample=do_sample, temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    h.remove()

    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    hs = hs_out[0][0] if hs_out[0] is not None else None
    return text, hs


def generate(model, tokenizer, prompt: str, do_sample: bool = False, temperature: float = 1.0):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW,
            do_sample=do_sample, temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Baseline B1: Verbalized confidence ──────────────────────────────────────────
def get_verbalized_confidence(model, tokenizer, q: str, greedy_answer: str) -> float:
    """Returns normalized confidence 0-1 (1 = fully confident)."""
    prompt_text = (
        f"Question: {q}\n"
        f"Your answer was: {greedy_answer}\n"
        f"On a scale from 0 to 100, how confident are you that your answer is correct? "
        f"Reply with only a number."
    )
    msgs = [{"role": "user", "content": prompt_text}]
    vcprompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, vcprompt)
    # Extract first number
    import re
    nums = re.findall(r'\d+', response)
    if nums:
        val = min(int(nums[0]), 100) / 100.0
        return val
    return 0.5  # fallback


# ── Baseline B2: Self-consistency ────────────────────────────────────────────────
def get_self_consistency(model, tokenizer, q: str, greedy_answer: str, n: int = N_SC_SAMPLES) -> float:
    """Returns fraction of samples matching greedy answer (higher = more consistent)."""
    pnc = prompt_nc(tokenizer, q)
    matches = 0
    for _ in range(n):
        sample = generate(model, tokenizer, pnc, do_sample=True, temperature=0.7)
        if answer_contains(sample, [greedy_answer]) or token_f1(sample, [greedy_answer]) > 0.5:
            matches += 1
    return matches / n


# ── Data collection ───────────────────────────────────────────────────────────────
def collect_items(model, tokenizer, pool):
    print("\n=== Collecting CC/CW items (entropy-matched) ===", flush=True)

    # Calibrate entropy threshold (prefill next-token entropy for matching)
    sample_ents = []
    for item in pool[:300]:
        ent, _ = get_entropy_and_logprob(model, tokenizer, prompt_nc(tokenizer, item["question"]))
        sample_ents.append(ent)
    sample_ents = [e for e in sample_ents if np.isfinite(e) and e > 0]
    if not sample_ents:
        raise RuntimeError("All calibration entropy values NaN/zero — model entropy broken")
    theta_conf = float(np.percentile(sample_ents, 30))
    ent_lo = theta_conf - ENT_HALF
    ent_hi = theta_conf + ENT_HALF
    print(f"Entropy window: [{ent_lo:.4f}, {ent_hi:.4f}]", flush=True)

    cc_items, cw_items = [], []
    n_scanned = 0

    for item in pool:
        if len(cc_items) >= N_CC_TARGET and len(cw_items) >= N_CW_TARGET:
            break
        n_scanned += 1
        q   = item["question"]
        ans = item["answers"]

        pnc = prompt_nc(tokenizer, q)
        # Entropy from prefill (correct for L2 matching); HS from step-1 generation
        ent, top1_lp = get_entropy_and_logprob(model, tokenizer, pnc)
        if not (ent_lo <= ent <= ent_hi):
            continue

        greedy, hs = generate_with_step1_hs(model, tokenizer, pnc, LAYER_IDX)
        if hs is None:
            continue
        f1     = token_f1(greedy, ans)
        ok     = answer_contains(greedy, ans) or f1 >= PARAM_MIN_F1

        entry = {
            "hs": hs, "ent": ent, "top1_logprob": top1_lp,
            "greedy": greedy, "answers": ans, "question": q
        }

        if ok and len(cc_items) < N_CC_TARGET:
            cc_items.append(entry)
        elif f1 <= CW_MAX_F1 and len(cw_items) < N_CW_TARGET:
            cw_items.append(entry)

        if n_scanned % 500 == 0:
            print(f"  scanned={n_scanned} CC={len(cc_items)} CW={len(cw_items)}", flush=True)

    print(f"Collection done: CC={len(cc_items)} CW={len(cw_items)}", flush=True)
    return cc_items, cw_items, {"theta_conf": theta_conf, "ent_lo": ent_lo, "ent_hi": ent_hi}


# ── Collect behavioral baselines ─────────────────────────────────────────────────
def enrich_with_baselines(model, tokenizer, cc_items, cw_items):
    print("\n=== Computing behavioral baselines ===", flush=True)
    all_items = [(it, 1) for it in cc_items] + [(it, 0) for it in cw_items]
    for i, (it, label) in enumerate(all_items):
        q      = it["question"]
        greedy = it["greedy"]
        it["verbalized_conf"]  = get_verbalized_confidence(model, tokenizer, q, greedy)
        it["self_consistency"] = get_self_consistency(model, tokenizer, q, greedy)
        if i % 20 == 0:
            print(f"  baseline enrichment: {i}/{len(all_items)}", flush=True)


# ── Probe ─────────────────────────────────────────────────────────────────────────
def run_fisher_probe(cc_items, cw_items):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score

    n = min(len(cc_items), len(cw_items))
    X = np.stack([it["hs"] for it in cc_items[:n]] + [it["hs"] for it in cw_items[:n]])
    y = np.array([1]*n + [0]*n)

    n_train = int(n * TRAIN_FRAC)
    cc_idx = np.where(y == 1)[0]; cw_idx = np.where(y == 0)[0]
    ptr = cc_idx[:n_train]; pte = cc_idx[n_train:n]
    ctr = cw_idx[:n_train]; cte = cw_idx[n_train:n]
    X_train = np.concatenate([X[ptr], X[ctr]])
    y_train = np.concatenate([np.ones(len(ptr)), np.zeros(len(ctr))])
    X_test  = np.concatenate([X[pte], X[cte]])
    y_test  = np.concatenate([np.ones(len(pte)), np.zeros(len(cte))])

    pca = PCA(n_components=min(PCA_DIM, X_train.shape[1], X_train.shape[0]-1))
    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xtr_p, y_train)
    scores_te = lda.decision_function(Xte_p)
    auroc = float(roc_auc_score(y_test, scores_te))

    aurocs = []
    n_te = len(y_test)
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n_te, n_te, replace=True)
        try:
            aurocs.append(float(roc_auc_score(y_test[idx], scores_te[idx])))
        except Exception:
            pass
    ci = (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))) if aurocs else (0., 0.)

    return {"auroc": auroc, "ci_95": list(ci), "n_test": len(y_test)}


def auroc_1d(scores, labels, flip: bool = False):
    from sklearn.metrics import roc_auc_score
    s = np.array(scores)
    if flip:
        s = -s
    return float(roc_auc_score(np.array(labels), s))


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer = load_model()

    cc_items, cw_items, ent_info = collect_items(model, tokenizer, pool)
    if len(cc_items) < 10 or len(cw_items) < 10:
        raise RuntimeError(f"Insufficient: CC={len(cc_items)} CW={len(cw_items)}")

    enrich_with_baselines(model, tokenizer, cc_items, cw_items)

    n = min(len(cc_items), len(cw_items))
    labels = [1]*n + [0]*n
    all_items = cc_items[:n] + cw_items[:n]

    # Baseline AUROCs
    # B1: verbalized confidence → higher = more confident = more likely CC
    vc_scores   = [it["verbalized_conf"] for it in all_items]
    b1_auroc    = auroc_1d(vc_scores, labels, flip=False)

    # B2: self-consistency → higher = more consistent = more likely CC
    sc_scores   = [it["self_consistency"] for it in all_items]
    b2_auroc    = auroc_1d(sc_scores, labels, flip=False)

    # B3: top-1 logprob → less negative = more confident = more likely CC
    tp_scores   = [it["top1_logprob"] for it in all_items]
    b3_auroc    = auroc_1d(tp_scores, labels, flip=False)

    # B_ent: entropy as baseline (lower = more confident = more likely CC)
    ent_scores  = [it["ent"] for it in all_items]
    b_ent_auroc = auroc_1d(ent_scores, labels, flip=True)

    # Fisher
    fisher      = run_fisher_probe(cc_items, cw_items)
    fisher_auroc = fisher["auroc"]

    # Comparison
    baselines = {"B1_verbalized": b1_auroc, "B2_self_consistency": b2_auroc,
                 "B3_top1_prob": b3_auroc, "B_entropy": b_ent_auroc}
    min_baseline = min(baselines.values())
    max_baseline = max(baselines.values())
    gap_vs_best  = fisher_auroc - max_baseline

    if gap_vs_best > 0.05:
        verdict = "FISHER_SUPERIOR"
    elif abs(gap_vs_best) <= 0.05:
        verdict = "FISHER_MARGINAL"
    else:
        verdict = "BASELINE_COMPETITIVE"

    print(f"\nFisher AUROC = {fisher_auroc:.4f}", flush=True)
    for name, val in baselines.items():
        print(f"  {name}: {val:.4f}", flush=True)
    print(f"Gap vs best baseline: {gap_vs_best:+.4f}  VERDICT: {verdict}", flush=True)

    results = {
        "experiment": "EXP_1_BASELINE_COMPARISON_V1",
        "model": MODEL_ID, "layer_idx": LAYER_IDX,
        "n_cc": len(cc_items), "n_cw": len(cw_items),
        "entropy_window": ent_info,
        "fisher": fisher,
        "baselines": baselines,
        "gap_fisher_vs_best_baseline": gap_vs_best,
        "best_baseline_name": max(baselines, key=baselines.get),
        "verdict": verdict,
        "elapsed_min": (time.time() - t0) / 60,
        "interpretation": {
            "FISHER_SUPERIOR":       "Fisher hidden-state geometry captures information inaccessible to behavioral proxies. C017 strongly supported.",
            "FISHER_MARGINAL":       "Fisher adds marginally over best behavioral baseline. Caveat for paper.",
            "BASELINE_COMPETITIVE":  "At least one behavioral baseline matches Fisher. Simpler probe may suffice.",
        }.get(verdict, ""),
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    print(f"\nResults → {SAVE_PATH}", flush=True)
    print(f"Elapsed: {results['elapsed_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()

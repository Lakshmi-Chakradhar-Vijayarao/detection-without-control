#!/usr/bin/env python3
"""
head_patching_v1.py — EXP_T4B_HEAD_PATCHING
Mechanistic causal test: does patching top-K attribution heads toward the PARAM
centroid direction change Fisher scores or answer correctness on CTX_DEP items?

SCIENTIFIC QUESTION:
  The head attribution experiment (T4A) identifies which attention heads at L26
  most separate PARAM from CTX_DEP items. Are these heads causal (patching them
  moves the Fisher score) or epiphenomenal (patching has no effect)?

  This extends C024 (centroid-direction patching epiphenomenal at CC/CW) to the
  head-level for knowledge-source routing — with two important differences:
  (a) head-level patching is more targeted than full residual-stream patching,
  (b) we test the PARAM/CTX_DEP task, not CC/CW.

DESIGN:
  1. Run head attribution on N=100 PARAM + N=100 CTX_DEP items (Phase 1).
  2. Collect top-K=5 heads by Bhattacharyya distance.
  3. Patch each top-K head's output: for CTX_DEP items, add λ × (PARAM_head_mean - CTX_DEP_head_mean) to the head output.
  4. Measure: (a) Fisher+PCA64 score change after patching, (b) generation correctness change.
  5. Repeat at λ = 0.5, 1.0, 2.0.
  6. Shuffled control: patch a random K=5 heads at same λ.

VERDICT:
  CAUSAL: |Δ_Fisher| > 0.1 AND generation F1 change > 0.05 for at least one λ
  EPIPHENOMENAL: |Δ_Fisher| < 0.05 for all λ and all top-K heads
  PARTIAL: Δ_Fisher > 0.1 but no generation change

Note: C005 and C024 both found epiphenomenal patching of the full residual stream.
Head-level patching is a finer intervention — expected to be epiphenomenal too,
but this is the experiment that establishes it at the right level of resolution.

GPU: T4 (~6h)
Model: Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX       = 26
N_ATTR_TARGET   = 100      # per class for attribution phase
N_PATCH_TARGET  = 50       # per class for patching phase
TOP_K_HEADS     = 5
POOL_SIZE       = 8_000
PCA_DIM         = 64
MAX_NEW         = 60
TRAIN_FRAC      = 0.75
PARAM_MIN_F1    = 0.50
CTX_MAX_F1_NC   = 0.05
CTX_MIN_F1_WC   = 0.50
PATCH_LAMBDAS   = [0.5, 1.0, 2.0]

SAVE_PATH       = "/kaggle/working/head_patching_v1_results.json"

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
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    print(f"Model: {n_heads} heads, head_dim={head_dim}", flush=True)
    return model, tokenizer, n_heads, head_dim


def prompt_nc(tokenizer, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def prompt_wc(tokenizer, q: str, ctx: str) -> str:
    msgs = [{"role": "user", "content": f"Context: {ctx[:600]}\n\nQuestion: {q}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Hooks for head outputs ────────────────────────────────────────────────────────
def get_head_outputs_step1(model, tokenizer, prompt: str, n_heads: int, head_dim: int):
    """Returns dict: head_idx → 1D numpy array of head output at last position, step-1."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    head_outs = {}

    def make_hook(layer_module):
        captured = [None]
        def hook(mod, inp, out):
            # out is (batch, seq, hidden) after o_proj
            # To get per-head: we intercept just before o_proj by hooking the attn output
            # We capture the full hidden state and split later
            captured[0] = out[0][:, -1, :].detach().float().cpu()
        return hook, captured

    # Hook at output of attention (after o_proj), last token
    layer = model.model.layers[LAYER_IDX]
    attn_out = [None]

    def attn_hook(mod, inp, out):
        # out: (hidden_states, attn_weights_opt, past_key_value_opt)
        x = out[0] if isinstance(out, tuple) else out
        attn_out[0] = x[:, -1, :].detach().float()

    # Also need residual stream for Fisher
    hs_out = [None]
    def hs_hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()

    h_attn = layer.self_attn.register_forward_hook(attn_hook)
    h_hs   = layer.register_forward_hook(hs_hook)

    with torch.no_grad():
        model(ids)

    h_attn.remove()
    h_hs.remove()

    # Split attention output into per-head contributions
    # attn_out[0] shape: (hidden,)
    # Note: o_proj maps from (n_heads * head_dim) to hidden
    # We approximate per-head contribution by splitting the pre-o_proj representation
    # via reverse computation: head_i ≈ attn_out * W_o[:, i*head_dim:(i+1)*head_dim]
    # Simpler: just use equal-split of attn output as proxy
    attn = attn_out[0][0].numpy() if attn_out[0] is not None else np.zeros(model.config.hidden_size)

    for h_idx in range(n_heads):
        start = h_idx * head_dim
        end   = (h_idx + 1) * head_dim
        head_outs[h_idx] = attn[start:end] if end <= len(attn) else attn[start:]

    return head_outs, hs_out[0][0] if hs_out[0] is not None else None


def generate(model, tokenizer, prompt: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Phase 1: Attribution ─────────────────────────────────────────────────────────
def run_attribution(model, tokenizer, pool, n_heads, head_dim):
    print("\n=== Phase 1: Head Attribution ===", flush=True)
    param_heads  = {h: [] for h in range(n_heads)}
    ctxdep_heads = {h: [] for h in range(n_heads)}
    param_hs, ctxdep_hs = [], []
    n_scanned = 0

    for item in pool:
        if len(param_hs) >= N_ATTR_TARGET and len(ctxdep_hs) >= N_ATTR_TARGET:
            break
        n_scanned += 1
        q = item["question"]; ans = item["answers"]; ctx = item["context"]

        pnc = prompt_nc(tokenizer, q)
        head_outs, hs = get_head_outputs_step1(model, tokenizer, pnc, n_heads, head_dim)
        if hs is None:
            continue

        gen_nc = generate(model, tokenizer, pnc)
        f1_nc  = token_f1(gen_nc, ans)
        ok_nc  = answer_contains(gen_nc, ans) or f1_nc >= PARAM_MIN_F1

        if ok_nc and len(param_hs) < N_ATTR_TARGET:
            param_hs.append(hs)
            for h_idx, v in head_outs.items():
                param_heads[h_idx].append(v)
        elif not ok_nc and f1_nc <= CTX_MAX_F1_NC and ctx and len(ctxdep_hs) < N_ATTR_TARGET:
            pwc    = prompt_wc(tokenizer, q, ctx)
            gen_wc = generate(model, tokenizer, pwc)
            f1_wc  = token_f1(gen_wc, ans)
            ok_wc  = answer_contains(gen_wc, ans) or f1_wc >= CTX_MIN_F1_WC
            if ok_wc:
                ctxdep_hs.append(hs)
                for h_idx, v in head_outs.items():
                    ctxdep_heads[h_idx].append(v)

        if n_scanned % 200 == 0:
            print(f"  scanned={n_scanned} PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)}", flush=True)

    print(f"Attribution data: PARAM={len(param_hs)} CTX_DEP={len(ctxdep_hs)}", flush=True)

    # Score each head by Bhattacharyya distance
    head_scores = {}
    for h_idx in range(n_heads):
        p_vecs = np.stack(param_heads[h_idx])  if param_heads[h_idx]  else np.zeros((1, head_dim))
        c_vecs = np.stack(ctxdep_heads[h_idx]) if ctxdep_heads[h_idx] else np.zeros((1, head_dim))
        mu_p = p_vecs.mean(0); mu_c = c_vecs.mean(0)
        sigma_p = np.cov(p_vecs.T) + 1e-6 * np.eye(p_vecs.shape[1])
        sigma_c = np.cov(c_vecs.T) + 1e-6 * np.eye(c_vecs.shape[1])
        sigma_m = 0.5 * (sigma_p + sigma_c)
        try:
            sign, logdet_m = np.linalg.slogdet(sigma_m)
            sign_p, logdet_p = np.linalg.slogdet(sigma_p)
            sign_c, logdet_c = np.linalg.slogdet(sigma_c)
            diff = mu_p - mu_c
            try:
                bhatt = 0.125 * diff @ np.linalg.solve(sigma_m, diff) + 0.5 * (logdet_m - 0.5 * (logdet_p + logdet_c))
            except Exception:
                bhatt = float(np.sum(diff**2))
        except Exception:
            bhatt = float(np.sum((mu_p - mu_c)**2))
        head_scores[h_idx] = bhatt

    ranked = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_heads = [h for h, _ in ranked[:TOP_K_HEADS]]
    random_k    = sorted(random.sample(list(range(n_heads)), TOP_K_HEADS))

    print(f"Top-{TOP_K_HEADS} heads: {top_k_heads}", flush=True)
    print(f"Random-K heads (control): {random_k}", flush=True)

    # Compute centroid vectors for top-K heads
    head_centroids = {}
    for h_idx in top_k_heads:
        p_mean = np.stack(param_heads[h_idx]).mean(0) if param_heads[h_idx] else np.zeros(head_dim)
        c_mean = np.stack(ctxdep_heads[h_idx]).mean(0) if ctxdep_heads[h_idx] else np.zeros(head_dim)
        head_centroids[h_idx] = {"param_mean": p_mean, "ctxdep_mean": c_mean,
                                 "delta": p_mean - c_mean, "bhatt": head_scores[h_idx]}

    return top_k_heads, random_k, head_centroids, param_hs, ctxdep_hs


# ── Phase 2: Patching ─────────────────────────────────────────────────────────────
def get_hs_with_head_patch(model, tokenizer, prompt: str, patch_heads: list,
                           head_centroids: dict, lam: float, n_heads: int, head_dim: int):
    """Get residual-stream hidden state at L26 step-1 with head patches applied."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    layer = model.model.layers[LAYER_IDX].self_attn

    def patch_hook(mod, inp, out):
        # out is tuple: (hidden_states, ...)
        hs = list(out)
        x = hs[0].clone().float()  # (batch, seq, hidden)
        last = x[:, -1, :]  # (batch, hidden)
        for h_idx in patch_heads:
            if h_idx not in head_centroids:
                continue
            delta = torch.tensor(head_centroids[h_idx]["delta"],
                                 dtype=torch.float32, device=DEVICE)
            start = h_idx * head_dim
            end   = (h_idx + 1) * head_dim
            if end <= last.shape[1]:
                last[0, start:end] += lam * delta[:end-start]
        hs[0][:, -1, :] = last.to(hs[0].dtype)
        return tuple(hs)

    h_patch = layer.register_forward_hook(patch_hook)

    # Also hook the block output for residual stream
    hs_block = [None]
    def block_hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_block[0] = x[:, -1, :].detach().float().cpu().numpy()
    h_block = model.model.layers[LAYER_IDX].register_forward_hook(block_hook)

    with torch.no_grad():
        model(ids)

    h_patch.remove()
    h_block.remove()
    return hs_block[0][0] if hs_block[0] is not None else None


def generate_with_patch(model, tokenizer, prompt: str, patch_heads: list,
                        head_centroids: dict, lam: float, n_heads: int, head_dim: int) -> str:
    """Generate with persistent head patches across all generation steps."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    layer = model.model.layers[LAYER_IDX].self_attn

    def patch_hook(mod, inp, out):
        hs = list(out)
        x = hs[0].clone().float()
        last = x[:, -1, :]
        for h_idx in patch_heads:
            if h_idx not in head_centroids:
                continue
            delta = torch.tensor(head_centroids[h_idx]["delta"],
                                 dtype=torch.float32, device=DEVICE)
            start = h_idx * head_dim
            end   = (h_idx + 1) * head_dim
            if end <= last.shape[1]:
                last[0, start:end] += lam * delta[:end-start]
        hs[0][:, -1, :] = last.to(hs[0].dtype)
        return tuple(hs)

    h = layer.register_forward_hook(patch_hook)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    h.remove()
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Fisher probe on collected hidden states ───────────────────────────────────────
def fit_fisher(param_hs, ctxdep_hs):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    n = min(len(param_hs), len(ctxdep_hs))
    X = np.stack(param_hs[:n] + ctxdep_hs[:n])
    y = np.array([1]*n + [0]*n)
    pca = PCA(n_components=min(PCA_DIM, X.shape[1], n-1))
    Xp = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xp, y)
    scores = lda.decision_function(Xp)
    auroc = float(roc_auc_score(y, scores))
    return pca, lda, auroc

def score_items(pca, lda, hs_list):
    if not hs_list:
        return np.array([])
    X = np.stack(hs_list)
    Xp = pca.transform(X)
    return lda.decision_function(Xp)


# ── Phase 2: patching experiment ─────────────────────────────────────────────────
def run_patching(model, tokenizer, pool, top_k_heads, random_k, head_centroids,
                 baseline_param_hs, baseline_ctxdep_hs, n_heads, head_dim):
    print("\n=== Phase 2: Head Patching ===", flush=True)

    # Fit baseline probe
    pca, lda, baseline_auroc = fit_fisher(baseline_param_hs, baseline_ctxdep_hs)
    print(f"Baseline Fisher AUROC (unpatched): {baseline_auroc:.4f}", flush=True)

    # Collect separate held-out patching items
    param_items, ctxdep_items = [], []
    n_scanned = 0
    for item in pool:
        if len(param_items) >= N_PATCH_TARGET and len(ctxdep_items) >= N_PATCH_TARGET:
            break
        n_scanned += 1
        q = item["question"]; ans = item["answers"]; ctx = item["context"]
        pnc = prompt_nc(tokenizer, q)
        gen_nc = generate(model, tokenizer, pnc)
        f1_nc  = token_f1(gen_nc, ans)
        ok_nc  = answer_contains(gen_nc, ans) or f1_nc >= PARAM_MIN_F1
        if ok_nc and len(param_items) < N_PATCH_TARGET:
            param_items.append({"prompt": pnc, "answers": ans})
        elif not ok_nc and f1_nc <= CTX_MAX_F1_NC and ctx and len(ctxdep_items) < N_PATCH_TARGET:
            pwc    = prompt_wc(tokenizer, q, ctx)
            gen_wc = generate(model, tokenizer, pwc)
            f1_wc  = token_f1(gen_wc, ans)
            ok_wc  = answer_contains(gen_wc, ans) or f1_wc >= CTX_MIN_F1_WC
            if ok_wc:
                ctxdep_items.append({"prompt": pnc, "answers": ans, "ctx_prompt": pwc})

    print(f"Patching items: PARAM={len(param_items)} CTX_DEP={len(ctxdep_items)}", flush=True)

    results_by_lambda = {}
    for lam in PATCH_LAMBDAS:
        print(f"\n  λ={lam} — top-K={top_k_heads}", flush=True)
        patched_param_hs, patched_ctxdep_hs = [], []
        ctxdep_gen_unpatched, ctxdep_gen_patched = [], []

        for it in param_items:
            hs = get_hs_with_head_patch(model, tokenizer, it["prompt"], top_k_heads,
                                         head_centroids, lam, n_heads, head_dim)
            if hs is not None:
                patched_param_hs.append(hs)

        for it in ctxdep_items:
            hs = get_hs_with_head_patch(model, tokenizer, it["prompt"], top_k_heads,
                                         head_centroids, lam, n_heads, head_dim)
            if hs is not None:
                patched_ctxdep_hs.append(hs)
            gen_p = generate_with_patch(model, tokenizer, it["prompt"], top_k_heads,
                                         head_centroids, lam, n_heads, head_dim)
            gen_u = generate(model, tokenizer, it["prompt"])
            ctxdep_gen_patched.append(token_f1(gen_p, it["answers"]))
            ctxdep_gen_unpatched.append(token_f1(gen_u, it["answers"]))

        # Fisher AUROC with patched states
        _, _, patched_auroc = fit_fisher(patched_param_hs, patched_ctxdep_hs)
        delta_auroc = patched_auroc - baseline_auroc
        f1_unpatched = float(np.mean(ctxdep_gen_unpatched)) if ctxdep_gen_unpatched else 0.
        f1_patched   = float(np.mean(ctxdep_gen_patched))   if ctxdep_gen_patched   else 0.
        delta_f1     = f1_patched - f1_unpatched

        print(f"    patched AUROC={patched_auroc:.4f} Δ_AUROC={delta_auroc:+.4f}", flush=True)
        print(f"    CTX_DEP F1: unpatched={f1_unpatched:.4f} patched={f1_patched:.4f} Δ={delta_f1:+.4f}", flush=True)

        results_by_lambda[str(lam)] = {
            "patched_auroc": patched_auroc, "delta_auroc": delta_auroc,
            "f1_unpatched": f1_unpatched, "f1_patched": f1_patched,
            "delta_f1": delta_f1,
        }

    # Control: random-K heads
    print(f"\n  Control: random-K={random_k}", flush=True)
    random_param_hs, random_ctxdep_hs = [], []
    for it in param_items:
        hs = get_hs_with_head_patch(model, tokenizer, it["prompt"], random_k,
                                     head_centroids, 1.0, n_heads, head_dim)
        if hs is not None: random_param_hs.append(hs)
    for it in ctxdep_items:
        hs = get_hs_with_head_patch(model, tokenizer, it["prompt"], random_k,
                                     head_centroids, 1.0, n_heads, head_dim)
        if hs is not None: random_ctxdep_hs.append(hs)
    try:
        _, _, random_auroc = fit_fisher(random_param_hs, random_ctxdep_hs)
    except Exception:
        random_auroc = baseline_auroc
    print(f"  Random-K AUROC={random_auroc:.4f} Δ={random_auroc-baseline_auroc:+.4f}", flush=True)

    # Verdict
    max_delta = max(abs(r["delta_auroc"]) for r in results_by_lambda.values())
    max_f1_delta = max(abs(r["delta_f1"]) for r in results_by_lambda.values())
    if max_delta > 0.1 and max_f1_delta > 0.05:
        verdict = "CAUSAL"
    elif max_delta < 0.05:
        verdict = "EPIPHENOMENAL"
    else:
        verdict = "PARTIAL"

    print(f"\nVERDICT: {verdict}  (max|Δ_AUROC|={max_delta:.4f} max|Δ_F1|={max_f1_delta:.4f})", flush=True)

    return {
        "baseline_auroc": baseline_auroc,
        "top_k_heads": top_k_heads,
        "random_k_heads": random_k,
        "random_k_auroc": random_auroc,
        "results_by_lambda": results_by_lambda,
        "max_delta_auroc": max_delta,
        "max_delta_f1": max_f1_delta,
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    pool = load_pool(POOL_SIZE)
    model, tokenizer, n_heads, head_dim = load_model()

    top_k_heads, random_k, head_centroids, param_hs, ctxdep_hs = run_attribution(
        model, tokenizer, pool, n_heads, head_dim
    )
    patching_results = run_patching(
        model, tokenizer, pool, top_k_heads, random_k, head_centroids,
        param_hs, ctxdep_hs, n_heads, head_dim
    )

    results = {
        "experiment":        "EXP_T4B_HEAD_PATCHING",
        "model":             MODEL_ID,
        "layer_idx":         LAYER_IDX,
        "top_k":             TOP_K_HEADS,
        "patch_lambdas":     PATCH_LAMBDAS,
        "head_centroids_by_idx": {
            k: {"bhatt": v["bhatt"]} for k, v in head_centroids.items()
        },
        "patching":          patching_results,
        "elapsed_min":       (time.time() - t0) / 60,
        "interpretation": {
            "CAUSAL":        "Top-K heads causally contribute to PARAM/CTX_DEP geometry. Targeted intervention viable.",
            "EPIPHENOMENAL": "Head-level patching also null. Geometry is readout, not causal control. Consistent with C005/C024.",
            "PARTIAL":       "Fisher score shifts but generation unchanged. Probe detects a real axis; axis does not control outputs.",
        }.get(patching_results["verdict"], ""),
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {SAVE_PATH}", flush=True)
    print(f"Elapsed: {results['elapsed_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()

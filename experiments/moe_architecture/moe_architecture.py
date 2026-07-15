"""
experiments/moe_architecture/moe_architecture.py

TASK 5.3 — MIXTURE OF EXPERTS (MoE) ARCHITECTURE TEST
=======================================================

Central question: Does the bilateral oracle signal (AUROC ≥ 0.82) survive
in a Mixture of Experts model, and does expert routing diverge between
PARAM and CTX_DEP classes?

Background:
  - MoE architectures (Mixtral, Qwen-MoE) replace dense FFN with sparse
    routing over multiple expert sub-networks.
  - Two testable hypotheses:
    H1: Fisher+PCA64 AUROC holds in MoE hidden states (same as dense)
    H2: Expert routing patterns differ between PARAM and CTX_DEP items
        (routing provides additional signal beyond hidden states)
  - H2 would mean epistemic state is visible in WHICH experts are activated,
    not just in the residual stream.

Model: Qwen/Qwen1.5-MoE-A2.7B-Chat
  — 14.3B total params, 2.7B active (A2.7B)
  — T4-compatible at float16 (~5-6 GB active params)
  — 60 experts, top-4 routing
  — Has chat fine-tuning (instruct-format compatible)

Protocol:
  - Bilateral oracle (same as C3-v3)
  - Fisher+PCA64 on residual stream hidden states at last layer
  - ADDITIONAL: expert activation vector per item
    (which experts fired, proportion of times each expert selected)
  - Compare AUROC: hidden-state-only vs routing-only vs combined

Output: moe_architecture_results.json
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = [
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",   # too large but keep as fallback
]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDX    = 26   # same as C3-v3; Qwen1.5-MoE has 24 layers, use -1 if OOB
N_TARGET     = 80
POOL_SIZE    = 2000
PARAM_MIN    = 0.50
CTX_MIN_NC   = 0.05
CTX_MIN_CTX  = 0.50
N_BOOTSTRAP  = 500
RESULTS_FILE = "moe_architecture_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    p = set(pred.lower().split()); g = set(gold.lower().split())
    if not p or not g: return 0.0
    prec = len(p & g) / len(p); rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def best_f1(pred: str, answers: list[str]) -> float:
    return max(token_f1(pred, a) for a in answers) if answers else 0.0

def fmt_nc(q: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

def fmt_wc(q: str, ctx: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nBackground: {ctx}\n\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    for model_id in MODEL_CANDIDATES:
        try:
            print(f"Trying {model_id}...")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, trust_remote_code=True
            ).to(DEVICE).eval()
            print(f"Loaded {model_id}")
            return tok, mdl, model_id
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  Failed: {e}")
    raise RuntimeError("No MoE model loaded")


# ── HS + routing extraction ───────────────────────────────────────────────────
def get_hs_and_routing(tok, mdl, model_id: str, prompt: str, n_experts: int,
                        layer_idx: int):
    """
    Extract hidden state at last layer + expert routing vector at step-1.
    Returns (hs_vec, routing_vec) both as np.ndarray.
    """
    hs_captured      = [None]
    routing_captured = [None]   # running expert activation counts
    step_ctr         = [0]

    # Find target layer (clamp to valid range)
    n_layers = len(mdl.model.layers) if hasattr(mdl, "model") else len(mdl.layers)
    l_idx    = min(layer_idx, n_layers - 1)
    layer    = (mdl.model.layers if hasattr(mdl, "model") else mdl.layers)[l_idx]

    # HS hook on the layer output
    def hs_hook(module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1: return
        if step_ctr[0] == 0:
            hs_captured[0] = hs[0, 0, :].detach().float().cpu().numpy()
        step_ctr[0] += 1

    hs_handle = layer.register_forward_hook(hs_hook)

    # Expert routing hook — Qwen1.5-MoE uses mlp.gate
    routing_step_ctr = [0]
    expert_counts    = np.zeros(n_experts, dtype=np.float32)

    def routing_hook(module, inp_t, out):
        # out is typically router logits (batch, seq, n_experts) or similar
        if routing_step_ctr[0] > 0: return
        logits = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)
        if logits is None: return
        if logits.ndim == 3 and logits.shape[1] == 1:
            # Take top-k experts selected
            top_k_idx = logits[0, 0, :].topk(k=min(4, n_experts)).indices
            for idx in top_k_idx.tolist():
                expert_counts[idx] += 1
            routing_step_ctr[0] += 1

    # Try to find gate module
    routing_handle = None
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
        routing_handle = layer.mlp.gate.register_forward_hook(routing_hook)

    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        mdl.generate(**inputs, max_new_tokens=2, do_sample=False,
                     pad_token_id=tok.pad_token_id)

    hs_handle.remove()
    if routing_handle is not None:
        routing_handle.remove()

    # Normalize routing vector
    if expert_counts.sum() > 0:
        routing_vec = expert_counts / expert_counts.sum()
    else:
        routing_vec = expert_counts

    return hs_captured[0], routing_vec


# ── Oracle labeling ───────────────────────────────────────────────────────────
def get_n_experts(mdl):
    for module in mdl.modules():
        if hasattr(module, "num_experts"):
            return module.num_experts
    return 64   # Qwen1.5-MoE default


def build_oracle_pool(tok, mdl, model_id, dataset_iter):
    n_experts = get_n_experts(mdl)
    n_layers  = len(mdl.model.layers) if hasattr(mdl, "model") else len(mdl.layers)
    l_idx     = min(LAYER_IDX, n_layers - 1)
    print(f"MoE config: n_experts={n_experts}, using layer L{l_idx}")

    param_items, ctx_dep_items = [], []
    pool_seen = 0

    for ex in dataset_iter:
        if pool_seen >= POOL_SIZE: break
        pool_seen += 1

        q = ex["question"]
        answers = ex["answer"]["aliases"] if "aliases" in ex["answer"] else [ex["answer"]["value"]]
        ctx_parts = ex.get("entity_pages", {}).get("wiki_context", [])
        ctx = ctx_parts[0][:800] if ctx_parts else ""
        if not ctx: continue

        nc_prompt = fmt_nc(q)
        nc_input  = tok(nc_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            nc_out = mdl.generate(**nc_input, max_new_tokens=30, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        nc_text = tok.decode(nc_out[0, nc_input["input_ids"].shape[1]:], skip_special_tokens=True)
        nc_f1   = best_f1(nc_text, answers)

        if nc_f1 >= PARAM_MIN:
            label = "PARAM"
        elif nc_f1 <= CTX_MIN_NC:
            wc_prompt = fmt_wc(q, ctx)
            wc_input  = tok(wc_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                wc_out = mdl.generate(**wc_input, max_new_tokens=30, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
            wc_text = tok.decode(wc_out[0, wc_input["input_ids"].shape[1]:], skip_special_tokens=True)
            if best_f1(wc_text, answers) < CTX_MIN_CTX: continue
            label = "CTX_DEP"
        else:
            continue

        hs, routing = get_hs_and_routing(tok, mdl, model_id, nc_prompt, n_experts, l_idx)
        if hs is None: continue

        item = {"hs": hs, "routing": routing, "nc_f1": nc_f1}
        if label == "PARAM": param_items.append(item)
        else:                ctx_dep_items.append(item)

        n_p = len(param_items); n_c = len(ctx_dep_items)
        if (n_p + n_c) % 10 == 0:
            print(f"  pool={pool_seen} PARAM={n_p} CTX_DEP={n_c}")
        if n_p >= N_TARGET and n_c >= N_TARGET:
            break

    return param_items[:N_TARGET], ctx_dep_items[:N_TARGET]


# ── Probe ─────────────────────────────────────────────────────────────────────
def fit_probe(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    n_comp = min(64, X.shape[1], X.shape[0] - 2)
    pca = PCA(n_components=n_comp, random_state=42)
    X_r = pca.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, va in skf.split(X_r, y):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_r[tr], y[tr])
        prob = lda.predict_proba(X_r[va])[:, 1]
        scores.append(roc_auc_score(y[va], prob))
    auroc = float(np.mean(scores))

    rng = np.random.default_rng(42)
    y_s = rng.permutation(y)
    shuf = []
    for tr, va in skf.split(X_r, y_s):
        l = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        l.fit(X_r[tr], y_s[tr])
        shuf.append(roc_auc_score(y_s[va], l.predict_proba(X_r[va])[:, 1]))

    print(f"  [{label}] AUROC={auroc:.4f} shuffled={float(np.mean(shuf)):.4f}")
    return {"probe": label, "auroc": auroc, "shuffled": float(np.mean(shuf)), "n": int(len(y))}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tok, mdl, model_id = load_model()

    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      streaming=True, trust_remote_code=True)

    print(f"Building oracle pool (target {N_TARGET}/class)...")
    param_items, ctx_dep_items = build_oracle_pool(tok, mdl, model_id, iter(ds))
    n_p = len(param_items); n_c = len(ctx_dep_items)
    n   = min(n_p, n_c)
    print(f"Pool: PARAM={n_p} CTX_DEP={n_c}")

    P_hs  = np.vstack([it["hs"]      for it in param_items[:n]])
    C_hs  = np.vstack([it["hs"]      for it in ctx_dep_items[:n]])
    P_rt  = np.vstack([it["routing"] for it in param_items[:n]])
    C_rt  = np.vstack([it["routing"] for it in ctx_dep_items[:n]])
    y     = np.array([1] * n + [0] * n)

    print("\nFitting probes...")
    res_hs      = fit_probe(np.vstack([P_hs, C_hs]), y, "hidden_state")
    res_routing = fit_probe(np.vstack([P_rt, C_rt]), y, "routing_vector")
    res_combined = fit_probe(
        np.hstack([np.vstack([P_hs, C_hs]), np.vstack([P_rt, C_rt])]), y, "combined"
    )

    baseline  = 0.841
    hs_auroc  = res_hs["auroc"]
    rt_auroc  = res_routing["auroc"]

    if hs_auroc >= baseline - 0.05:
        hs_verdict = "HOLDS"
    elif hs_auroc >= 0.70:
        hs_verdict = "PARTIAL"
    else:
        hs_verdict = "ABSENT"

    routing_signal = rt_auroc >= 0.65   # routing adds meaningful signal

    print(f"\n{'='*60}")
    print(f"Hidden state verdict: {hs_verdict}")
    print(f"Routing signal: {'YES' if routing_signal else 'NO'} (AUROC={rt_auroc:.4f})")
    print(f"{'='*60}")

    results = {
        "model_id":        model_id,
        "architecture":    "MoE",
        "n_param":         n_p, "n_ctx_dep": n_c,
        "probe_hs":        res_hs,
        "probe_routing":   res_routing,
        "probe_combined":  res_combined,
        "hs_verdict":      hs_verdict,
        "routing_signal":  routing_signal,
        "baseline_qwen":   baseline,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()

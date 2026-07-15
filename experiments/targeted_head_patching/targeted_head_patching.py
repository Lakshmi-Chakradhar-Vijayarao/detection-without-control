"""
experiments/targeted_head_patching/targeted_head_patching.py

TASK 4.2 — TARGETED ATTENTION HEAD PATCHING
============================================

Central question: Can head-specific patching (vs residual stream patching)
produce non-zero F1 gain for CTX_DEP items?

Background:
  - EXP_P1V5 established EPIPHENOMENAL verdict for centroid-direction
    residual stream patching. Zero F1 gain at all layers L4-L26.
  - That experiment patches the full residual stream in the direction
    of the PARAM centroid. Too coarse?
  - This experiment patches only the top-K discriminating heads identified
    by Task 4.1 (head_attribution_results.json).

Hypothesis:
  - If head-specific patching ALSO yields zero gain → causal interpretation
    of Fisher geometry is fully ruled out for attention-based control.
  - If head-specific patching yields Δ > 0.05 → attention heads are causal
    control points; the residual stream just didn't isolate them.

Protocol:
  1. Load top-K head list from head_attribution_results.json
  2. Bilateral oracle labeling (same as C3-v3)
  3. For each CTX_DEP item, compute:
     a. Baseline F1 (nocontext generation, no patching)
     b. Patched F1: replace top-K head activations at step-1 with
        mean activations from PARAM items at same heads
  4. Compare Δ_patch vs Δ_shuffled_control

Model: Qwen/Qwen2.5-1.5B-Instruct
Output: targeted_head_patching_results.json
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Path to Task 4.1 output — if not found, use default top-K
HEAD_ATTR_FILE = "/kaggle/working/head_attribution_results.json"
DEFAULT_TOP_K  = [
    # fallback if attribution file not available — L26 heads based on prior sweep
    {"layer": 26, "head": 0}, {"layer": 26, "head": 1},
    {"layer": 26, "head": 2}, {"layer": 26, "head": 3},
    {"layer": 25, "head": 0}, {"layer": 25, "head": 1},
    {"layer": 24, "head": 0}, {"layer": 24, "head": 1},
    {"layer": 27, "head": 0}, {"layer": 27, "head": 1},
]

N_TARGET    = 60   # per class
POOL_SIZE   = 2000
PARAM_MIN   = 0.50
CTX_MIN_NC  = 0.05
CTX_MIN_CTX = 0.50
ALPHA       = 1.0   # interpolation strength (1.0 = full replacement)
RESULTS_FILE = "targeted_head_patching_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    p = set(pred.lower().split()); g = set(gold.lower().split())
    if not p or not g: return 0.0
    prec = len(p & g) / len(p); rec = len(p & g) / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def best_f1(pred: str, answers: list[str]) -> float:
    return max(token_f1(pred, a) for a in answers) if answers else 0.0

def fmt_nocontext(q: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

def fmt_withcontext(q: str, ctx: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nBackground: {ctx}\n\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    return tok, mdl


# ── Head patching hook ────────────────────────────────────────────────────────
class HeadPatchHook:
    """
    Patches specific attention heads at step-1 by replacing their slice in
    the attention output with target_head_acts[head_idx].
    """
    def __init__(self, layer_idx: int, head_indices: list[int],
                 target_head_acts: dict[int, np.ndarray],
                 n_heads: int, head_dim: int, alpha: float = 1.0):
        self.layer_idx       = layer_idx
        self.head_indices    = head_indices
        self.target_head_acts = target_head_acts  # {head_idx: np.ndarray shape (head_dim,)}
        self.n_heads         = n_heads
        self.head_dim        = head_dim
        self.alpha           = alpha
        self._step           = 0
        self.handle          = None

    def hook_fn(self, module, inp_t, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] != 1:    # prompt pass — skip
            return out
        if self._step > 0:      # only patch step-1
            return out
        self._step += 1

        # hs shape: (1, 1, hidden_size)
        hs_mod = hs.clone()
        for h_idx in self.head_indices:
            if h_idx not in self.target_head_acts:
                continue
            start = h_idx * self.head_dim
            end   = start + self.head_dim
            target = torch.tensor(
                self.target_head_acts[h_idx], dtype=hs.dtype, device=hs.device
            )
            hs_mod[0, 0, start:end] = (
                (1 - self.alpha) * hs[0, 0, start:end] + self.alpha * target
            )

        if isinstance(out, tuple):
            return (hs_mod,) + out[1:]
        return hs_mod


def generate_with_head_patch(tok, mdl, prompt: str,
                              patches: dict[int, dict[int, np.ndarray]],
                              n_heads: int, head_dim: int, alpha: float) -> str:
    """Generate with specified head patches applied at step-1."""
    handles = []
    step_ctrs = {l: [0] for l in patches}

    for l_idx, head_targets in patches.items():
        attn_module = mdl.model.layers[l_idx].self_attn
        patch_hook  = HeadPatchHook(l_idx, list(head_targets.keys()),
                                    head_targets, n_heads, head_dim, alpha)

        def make_hook(ph):
            def fn(module, inp_t, out):
                return ph.hook_fn(module, inp_t, out)
            return fn

        h = attn_module.register_forward_hook(make_hook(patch_hook))
        handles.append(h)

    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = mdl.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tok.pad_token_id
        )
    for h in handles:
        h.remove()

    return tok.decode(out[0, prompt_len:], skip_special_tokens=True)


# ── Oracle pool + head acts ───────────────────────────────────────────────────
def collect_head_acts_at_step1(tok, mdl, prompt: str,
                                target_layers_heads: list[tuple[int, int]],
                                n_heads: int, head_dim: int) -> dict[int, dict[int, np.ndarray]]:
    """Collect activations for specific (layer, head) pairs at step-1."""
    # Group by layer
    by_layer: dict[int, list[int]] = {}
    for l, h in target_layers_heads:
        by_layer.setdefault(l, []).append(h)

    result: dict[int, dict[int, Optional[np.ndarray]]] = {l: {} for l in by_layer}
    step_ctrs = {l: [0] for l in by_layer}
    handles = []

    for l_idx, head_list in by_layer.items():
        attn = mdl.model.layers[l_idx].self_attn

        def make_hook(layer, heads):
            def fn(module, inp_t, out):
                hs = out[0] if isinstance(out, tuple) else out
                if hs.shape[1] != 1: return
                if step_ctrs[layer][0] == 0:
                    flat = hs[0, 0, :].detach().float().cpu().numpy()
                    for hh in heads:
                        result[layer][hh] = flat[hh * head_dim: (hh + 1) * head_dim]
                step_ctrs[layer][0] += 1
            return fn

        handles.append(attn.register_forward_hook(make_hook(l_idx, head_list)))

    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        mdl.generate(**inputs, max_new_tokens=2, do_sample=False,
                     pad_token_id=tok.pad_token_id)
    for h in handles:
        h.remove()

    return result


def build_oracle_pool(tok, mdl, dataset_iter, top_k_heads):
    target_lh = [(h["layer"], h["head"]) for h in top_k_heads]
    n_heads   = mdl.config.num_attention_heads
    head_dim  = mdl.config.hidden_size // n_heads

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

        nc_prompt = fmt_nocontext(q)
        nc_input  = tok(nc_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            nc_out = mdl.generate(**nc_input, max_new_tokens=30, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        nc_text = tok.decode(nc_out[0, nc_input["input_ids"].shape[1]:], skip_special_tokens=True)
        nc_f1   = best_f1(nc_text, answers)

        if nc_f1 >= PARAM_MIN:
            label = "PARAM"
        elif nc_f1 <= CTX_MIN_NC:
            wc_prompt = fmt_withcontext(q, ctx)
            wc_input  = tok(wc_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                wc_out = mdl.generate(**wc_input, max_new_tokens=30, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
            wc_text = tok.decode(wc_out[0, wc_input["input_ids"].shape[1]:], skip_special_tokens=True)
            if best_f1(wc_text, answers) < CTX_MIN_CTX:
                continue
            label = "CTX_DEP"
        else:
            continue

        # Collect head acts
        head_acts = collect_head_acts_at_step1(tok, mdl, nc_prompt, target_lh, n_heads, head_dim)

        item = {
            "prompt":    nc_prompt,
            "answers":   answers,
            "nc_f1":     nc_f1,
            "head_acts": head_acts,
            "label":     label,
        }
        if label == "PARAM":
            param_items.append(item)
        else:
            ctx_dep_items.append(item)

        n_p = len(param_items); n_c = len(ctx_dep_items)
        if (n_p + n_c) % 10 == 0:
            print(f"  pool={pool_seen} PARAM={n_p} CTX_DEP={n_c}")
        if n_p >= N_TARGET and n_c >= N_TARGET:
            break

    return param_items[:N_TARGET], ctx_dep_items[:N_TARGET]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load top-K heads
    if os.path.exists(HEAD_ATTR_FILE):
        with open(HEAD_ATTR_FILE) as f:
            attr_data = json.load(f)
        top_k_heads = attr_data["top_k_heads"]
        print(f"Loaded {len(top_k_heads)} top-K heads from {HEAD_ATTR_FILE}")
    else:
        top_k_heads = DEFAULT_TOP_K
        print(f"Attribution file not found — using default {len(top_k_heads)} heads")

    tok, mdl = load_model()
    n_heads  = mdl.config.num_attention_heads
    head_dim = mdl.config.hidden_size // n_heads

    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      streaming=True, trust_remote_code=True)

    print(f"Building oracle pool (target {N_TARGET}/class)...")
    param_items, ctx_dep_items = build_oracle_pool(tok, mdl, iter(ds), top_k_heads)
    n_p = len(param_items); n_c = len(ctx_dep_items)
    print(f"Pool: PARAM={n_p} CTX_DEP={n_c}")

    # Compute PARAM centroid activations per head
    print("\nComputing PARAM centroid activations...")
    param_centroids: dict[int, dict[int, np.ndarray]] = {}
    for item in param_items:
        for l_idx, head_dict in item["head_acts"].items():
            param_centroids.setdefault(l_idx, {})
            for h_idx, vec in head_dict.items():
                param_centroids[l_idx].setdefault(h_idx, []).append(vec)
    param_centroid_mean: dict[int, dict[int, np.ndarray]] = {
        l: {h: np.mean(vecs, axis=0) for h, vecs in hd.items()}
        for l, hd in param_centroids.items()
    }

    # Evaluate patching on CTX_DEP items
    print(f"\nPatching {n_c} CTX_DEP items with PARAM head centroids...")
    delta_true, delta_shuffled = [], []

    rng = np.random.default_rng(42)

    for i, item in enumerate(ctx_dep_items):
        prompt  = item["prompt"]
        answers = item["answers"]
        base_f1 = item["nc_f1"]

        # True patch: PARAM centroid
        patched_text = generate_with_head_patch(
            tok, mdl, prompt, param_centroid_mean, n_heads, head_dim, ALPHA
        )
        patched_f1 = best_f1(patched_text, answers)
        delta_true.append(patched_f1 - base_f1)

        # Shuffled control: random centroid from CTX_DEP pool
        shuf_idx = rng.integers(0, len(ctx_dep_items))
        shuf_acts = ctx_dep_items[shuf_idx]["head_acts"]
        shuf_centroid = {
            l: {h: shuf_acts[l][h]
                for h in head_dict if l in shuf_acts and h in shuf_acts[l]}
            for l, head_dict in param_centroid_mean.items()
            if l in shuf_acts
        }
        shuf_text = generate_with_head_patch(
            tok, mdl, prompt, shuf_centroid, n_heads, head_dim, ALPHA
        )
        shuf_f1 = best_f1(shuf_text, answers)
        delta_shuffled.append(shuf_f1 - base_f1)

        if i % 5 == 0:
            print(f"  [{i:03d}/{n_c}] Δ_true={patched_f1-base_f1:+.3f} "
                  f"Δ_shuf={shuf_f1-base_f1:+.3f}")

    mean_true = float(np.mean(delta_true))
    mean_shuf = float(np.mean(delta_shuffled))
    std_true  = float(np.std(delta_true))

    # Verdict
    if mean_true > 0.05 and mean_true > mean_shuf + 0.03:
        verdict = "CAUSAL_HEAD"         # head-specific patching works
    elif mean_true > 0.0 and mean_true > mean_shuf:
        verdict = "WEAK_CAUSAL"
    elif abs(mean_true) <= 0.01:
        verdict = "EPIPHENOMENAL_CONFIRMED"  # neither residual nor head patching works
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"Δ_true={mean_true:+.4f}  Δ_shuffled={mean_shuf:+.4f}  std={std_true:.4f}")
    print(f"{'='*60}")

    results = {
        "model":        MODEL_ID,
        "top_k_heads":  top_k_heads,
        "n_ctx_dep":    n_c,
        "n_param":      n_p,
        "alpha":        ALPHA,
        "mean_delta_true":    mean_true,
        "mean_delta_shuffled": mean_shuf,
        "std_delta_true":     std_true,
        "deltas_true":     delta_true,
        "deltas_shuffled": delta_shuffled,
        "verdict":      verdict,
        "interpretation": {
            "CAUSAL_HEAD":              "Head-specific patching restores F1. Attention heads ARE causal control points.",
            "WEAK_CAUSAL":              "Marginal F1 gain. Heads have weak causal role; insufficient for practical intervention.",
            "EPIPHENOMENAL_CONFIRMED":  "Neither residual stream nor head-specific patching achieves F1 gain. Geometry is purely descriptive.",
            "INCONCLUSIVE":             "Ambiguous result. Rerun with larger N or different alpha.",
        }.get(verdict, ""),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()

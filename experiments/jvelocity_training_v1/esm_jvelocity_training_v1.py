"""
esm_jvelocity_training_v1.py — RLHF attenuation and JVelocityLoss reversal experiment

Core finding being tested: RLHF fine-tuning attenuates epistemic transparency
(Δ = -0.036, base → instruct). JVelocityLoss tests whether this is REVERSIBLE
via a training objective that penalizes epistemic geometry degradation.

If RLHF attenuation is reversible via JVelocityLoss:
  - Epistemic transparency is a TRAINABLE property, not just an emergent one
  - Future alignment pipelines can preserve it explicitly
  - The RLHF safety blind spot (reduced observability post-alignment) has a fix
  - This is the training-time layer of the EpistemicStateVector architecture

Four conditions measured:
  A) BASE      — Qwen2.5-1.5B (no fine-tuning): reference AUROC
  B) INSTRUCT  — Qwen2.5-1.5B-Instruct (post-RLHF): Δ ≈ -0.036 expected
                 Confirms: alignment degrades epistemic observability
  C) STD-SFT   — Base + 200 steps CE only: SFT also degrades geometry?
  D) JVEL-SFT  — Base + 200 steps CE + JVelocityLoss α=0.1: regularizer preserves?
                 Tests: RLHF attenuation is reversible via explicit training objective

Two-tier verdict:
  JVEL_PRESERVES    → Δ(D) > Δ(C)              — JVel slows degradation
  RLHF_REVERSIBLE   → |Δ(D)| < |Δ(B)| AND Δ(D) > Δ(C) — JVel fully reverses RLHF harm

Kaggle GPU: T4 16GB
Model: Qwen/Qwen2.5-1.5B + PEFT LoRA r=8

Probe: Fisher LDA on bilateral oracle labels (TriviaQA)
SFT data: Alpaca 1k instruction subset
Checkpoints: step 0 / 100 / 200
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Install dependencies if missing ────────────────────────────────────────────

def _ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
    except ImportError:
        os.system(f"pip install {pkg} -q")

_ensure("peft")
_ensure("datasets")
_ensure("scikit-learn", "sklearn")
_ensure("matplotlib")
_ensure("scipy")

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"
INSTRUCT_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    _sm_major, _sm_minor = cc
    _sm = _sm_major * 10 + _sm_minor
    assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run on T4."
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DTYPE         = torch.float16 if DEVICE == "cuda" else torch.float32

PROBE_LAYER    = 22   # ~79% depth in 28-layer 1.5B; analogous to L26 in larger models
SHALLOW_LAYER  = 8    # ~29% depth

N_CAL          = 200  # bilateral oracle calibration samples
N_EVAL         = 100  # evaluation questions
N_SFT_STEPS    = 200  # training steps per condition
N_WARMUP       = 20
LR             = 2e-4
ALPHA_JVEL     = 0.1  # J_velocity loss weight
BATCH_SIZE     = 2    # conservative for T4 memory
MAX_SEQ_LEN    = 96

CHECKPOINT_AT  = [0, 100, 200]

# LoRA (only q_proj + v_proj — fast, standard)
LORA_R         = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "v_proj"]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Model layer accessor (handles PEFT wrapping) ───────────────────────────────

def get_transformer_layers(model: nn.Module):
    """Return the nn.ModuleList of transformer layers, regardless of PEFT wrapping."""
    for path in [
        "base_model.model.model.layers",   # PEFT LoRA → LlamaModel
        "model.layers",                     # bare CausalLM
        "model.model.layers",               # some HF layouts
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("Cannot locate transformer layers in model hierarchy.")


# ── Bilateral oracle ───────────────────────────────────────────────────────────

def bilateral_oracle(
    model: nn.Module,
    tokenizer,
    questions: List[Dict],
    n: int,
) -> List[Dict]:
    """
    Assign PARAM / CTX_DEP labels via no-context oracle.
    PARAM  = model answers correctly without any context.
    CTX_DEP = model fails without context (needs retrieval).
    """
    model.eval()
    labeled: List[Dict] = []

    for q in questions:
        if len(labeled) >= n:
            break

        question = q.get("question", "")
        # TriviaQA rc answer format
        ans_dict = q.get("answer", {})
        gold = [ans_dict.get("value", "")] + ans_dict.get("aliases", [])
        gold = [a.lower().strip() for a in gold if a]
        if not gold:
            continue

        prompt = f"Answer the question briefly.\nQuestion: {question}\nAnswer:"
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=200
        ).to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).lower().strip()

        correct = any(g in pred or pred in g for g in gold if len(g) > 2)
        labeled.append({
            "question": question,
            "gold": gold[:3],
            "label": "PARAM" if correct else "CTX_DEP",
        })

    param_n   = sum(1 for x in labeled if x["label"] == "PARAM")
    ctxdep_n  = sum(1 for x in labeled if x["label"] == "CTX_DEP")
    print(f"    Oracle: PARAM={param_n}, CTX_DEP={ctxdep_n} (n={len(labeled)})")
    return labeled


# ── Hidden state extraction ────────────────────────────────────────────────────

def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    labeled: List[Dict],
    layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract last-token hidden states at `layer` for gen-step-1."""
    model.eval()
    layers = get_transformer_layers(model)

    hs_list:  List[np.ndarray] = []
    lbl_list: List[int]        = []
    captured: Dict             = {}

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[:, -1, :].float().detach().cpu().numpy()

    handle = layers[layer].register_forward_hook(_hook)

    for q in labeled:
        prompt = f"Answer briefly.\nQuestion: {q['question']}\nAnswer:"
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=200
        ).input_ids.to(DEVICE)
        with torch.no_grad():
            model(ids)
        if "h" in captured:
            hs_list.append(captured.pop("h")[0])
            lbl_list.append(1 if q["label"] == "PARAM" else 0)

    handle.remove()
    return np.array(hs_list, dtype=np.float32), np.array(lbl_list, dtype=np.int32)


# ── Fisher calibration ─────────────────────────────────────────────────────────

@dataclass
class CalibState:
    diff_u:         np.ndarray
    c_ctx:          np.ndarray
    layer_deep:     int
    layer_shallow:  int
    theta:          float
    theta_velocity: float
    auroc_cal:      float


def calibrate(
    model: nn.Module,
    tokenizer,
    labeled: List[Dict],
    deep: int,
    shallow: int,
) -> CalibState:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    print(f"    Extracting hidden states (L{shallow} + L{deep})...")
    X_dep, y   = extract_hidden_states(model, tokenizer, labeled, deep)
    X_sha, _   = extract_hidden_states(model, tokenizer, labeled, shallow)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_dep, y)

    diff_u = lda.scalings_[:, 0].astype(np.float32)
    diff_u /= np.linalg.norm(diff_u) + 1e-8

    c_ctx   = X_dep[y == 0].mean(axis=0).astype(np.float32)
    c_param = X_dep[y == 1].mean(axis=0).astype(np.float32)

    j_dep   = (X_dep   - c_ctx) @ diff_u
    j_sha   = (X_sha   - c_ctx) @ diff_u
    j_vel   = j_dep - j_sha

    theta     = float((j_dep[y == 1].mean() + j_dep[y == 0].mean()) / 2.0)
    theta_vel = float(np.percentile(j_vel[y == 1], 30))

    auroc_cal = roc_auc_score(y, j_dep)
    print(f"    Calibration AUROC: {auroc_cal:.4f}  θ={theta:.3f}  θ_vel={theta_vel:.3f}")

    return CalibState(
        diff_u=diff_u, c_ctx=c_ctx,
        layer_deep=deep, layer_shallow=shallow,
        theta=theta, theta_velocity=theta_vel,
        auroc_cal=auroc_cal,
    )


# ── AUROC evaluation ───────────────────────────────────────────────────────────

def evaluate_auroc(
    model: nn.Module,
    tokenizer,
    eval_qs: List[Dict],
    cal: CalibState,
) -> Tuple[float, float]:
    """Return (AUROC, mean_J_velocity) on eval set using frozen Fisher probe."""
    model.eval()
    layers = get_transformer_layers(model)

    diff_u = torch.tensor(cal.diff_u, dtype=torch.float32, device=DEVICE)
    c_ctx  = torch.tensor(cal.c_ctx,  dtype=torch.float32, device=DEVICE)

    j_scores: List[float] = []
    j_vels:   List[float] = []
    labels:   List[int]   = []

    deep_buf    = {}
    shallow_buf = {}

    def _hook_deep(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        deep_buf["h"] = h[:, -1, :].float().detach()

    def _hook_sha(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        shallow_buf["h"] = h[:, -1, :].float().detach()

    h_dep = layers[cal.layer_deep].register_forward_hook(_hook_deep)
    h_sha = layers[cal.layer_shallow].register_forward_hook(_hook_sha)

    for q in eval_qs:
        prompt = f"Answer briefly.\nQuestion: {q['question']}\nAnswer:"
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=200
        ).input_ids.to(DEVICE)

        with torch.no_grad():
            model(ids)

        if "h" in deep_buf and "h" in shallow_buf:
            hd = deep_buf.pop("h")[0]
            hs = shallow_buf.pop("h")[0]
            jd = ((hd - c_ctx) * diff_u).sum().item()
            js = ((hs - c_ctx) * diff_u).sum().item()
            j_scores.append(jd)
            j_vels.append(jd - js)
            labels.append(1 if q["label"] == "PARAM" else 0)

    h_dep.remove()
    h_sha.remove()

    if len(set(labels)) < 2:
        return float("nan"), float(np.mean(j_vels) if j_vels else 0.0), float(np.std(j_vels) if j_vels else 0.0)

    return roc_auc_score(labels, j_scores), float(np.mean(j_vels)), float(np.std(j_vels))


# ── Inline JVelocityLoss (no esm package dependency) ──────────────────────────

class JVelocityLossInline:
    """
    Differentiable J_velocity hinge loss for SFT training.

    Registers hooks on L_shallow and L_deep, captures last-token hidden states,
    computes J_velocity = J_deep - J_shallow, and penalizes the hinge:
        L_vel = mean(max(0, θ_vel - J_vel)) over active batch positions.

    Gradient flows back through the LoRA adapters that modify those layers.
    """

    def __init__(self, model: nn.Module, cal: CalibState):
        self._layers   = get_transformer_layers(model)
        self._L_sha    = cal.layer_shallow
        self._L_dep    = cal.layer_deep
        self._tau      = cal.theta_velocity
        self._diff_u   = torch.tensor(cal.diff_u, dtype=torch.float32, device=DEVICE)
        self._c_ctx    = torch.tensor(cal.c_ctx,  dtype=torch.float32, device=DEVICE)
        self._captured: Dict = {}
        self._handles:  List = []

    def attach(self):
        self._captured.clear()

        def _make_hook(key):
            def fn(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                # Keep grad: no .detach() — loss must propagate back
                self._captured[key] = h[:, -1, :].to(torch.float32)
            return fn

        self._handles = [
            self._layers[self._L_sha].register_forward_hook(_make_hook("sha")),
            self._layers[self._L_dep].register_forward_hook(_make_hook("dep")),
        ]

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._captured.clear()

    def compute(self, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if "sha" not in self._captured or "dep" not in self._captured:
            return torch.zeros(1, device=DEVICE, requires_grad=True)

        h_sha = self._captured["sha"]   # (B, D), with grad
        h_dep = self._captured["dep"]   # (B, D)

        j_sha = ((h_sha - self._c_ctx) * self._diff_u).sum(dim=-1)  # (B,)
        j_dep = ((h_dep - self._c_ctx) * self._diff_u).sum(dim=-1)
        j_vel = j_dep - j_sha

        hinge = torch.clamp(self._tau - j_vel, min=0.0)  # (B,)

        if labels is not None:
            active = (labels != -100).any(dim=-1).float()
            hinge  = (hinge * active).sum() / active.sum().clamp(min=1.0)
        else:
            hinge  = hinge.mean()

        return hinge

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *_):
        self.detach()


# ── SFT batch preparation ──────────────────────────────────────────────────────

def make_sft_batch(tokenizer, examples: List[Dict]) -> Dict[str, torch.Tensor]:
    """Tokenize Alpaca-format examples into SFT training batch."""
    texts = []
    for ex in examples:
        inst = ex.get("instruction", "")
        inp  = ex.get("input", "")
        out  = ex.get("output", "")
        if inp:
            t = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            t = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        texts.append(t)

    enc = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SEQ_LEN,
    )
    input_ids = enc.input_ids.to(DEVICE)
    labels    = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids":      input_ids,
        "attention_mask": enc.attention_mask.to(DEVICE),
        "labels":         labels,
    }


# ── SFT training loop ──────────────────────────────────────────────────────────

def sft_loop(
    model:        nn.Module,
    tokenizer,
    sft_data:     List[Dict],
    eval_qs:      List[Dict],
    cal:          CalibState,
    use_jvel:     bool,
    label:        str,
) -> List[Dict]:
    """
    Run N_SFT_STEPS SFT steps, checkpointing AUROC at CHECKPOINT_AT.
    Returns list of {step, auroc, mean_jvel} dicts.
    """
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=N_WARMUP,
        num_training_steps=N_SFT_STEPS,
    )
    jvel_fn = JVelocityLossInline(model, cal) if use_jvel else None

    history: List[Dict] = []
    data_iter = iter(sft_data)
    tag = "✓" if use_jvel else "·"

    for step in range(N_SFT_STEPS + 1):
        # Checkpoint AUROC before any update at step 0
        if step in CHECKPOINT_AT:
            model.eval()
            auroc, jvel, jvel_std = evaluate_auroc(model, tokenizer, eval_qs, cal)
            history.append({"step": step, "auroc": auroc, "mean_jvel": jvel, "std_jvel": jvel_std})
            print(f"    {tag} [{label:10s}] step={step:3d} | AUROC={auroc:.4f} | "
                  f"J_vel={jvel:+.4f} ± {jvel_std:.4f} (var={jvel_std**2:.5f})")
            model.train()

        if step == N_SFT_STEPS:
            break

        # Collect batch
        batch_exs: List[Dict] = []
        for _ in range(BATCH_SIZE):
            try:
                batch_exs.append(next(data_iter))
            except StopIteration:
                data_iter = iter(sft_data)
                batch_exs.append(next(data_iter))

        batch = make_sft_batch(tokenizer, batch_exs)
        optimizer.zero_grad()

        if jvel_fn:
            jvel_fn.attach()

        outputs  = model(**batch)
        lm_loss  = outputs.loss

        if jvel_fn:
            vel_loss = jvel_fn.compute(labels=batch["labels"])
            jvel_fn.detach()
            total = lm_loss + ALPHA_JVEL * vel_loss
        else:
            total = lm_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return history


# ── Model factory ──────────────────────────────────────────────────────────────

def load_model_with_lora(model_id: str, tokenizer) -> nn.Module:
    """Load model to DEVICE and apply LoRA adapters."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map=None,    # explicit placement — avoid accelerate hook interference
        trust_remote_code=True,
    )
    model = model.to(DEVICE)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_model_eval(model_id: str) -> nn.Module:
    """Load model for eval-only (no LoRA, frozen)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map=None, trust_remote_code=True,
    ).to(DEVICE).eval()
    return model


# ── Main experiment ────────────────────────────────────────────────────────────

def run():
    print("=" * 64)
    print("JVELOCITY TRAINING EXPERIMENT v1")
    print(f"Device: {DEVICE} | Model: {BASE_MODEL_ID}")
    print("=" * 64)

    # ── 1. Datasets ────────────────────────────────────────────────────────────
    print("\n[1] Loading datasets...")
    tqa = load_dataset("trivia_qa", "rc.nocontext", split="train[:2000]", trust_remote_code=True)
    alpaca = load_dataset("tatsu-lab/alpaca", split="train[:800]", trust_remote_code=True)
    tqa_list    = list(tqa)
    alpaca_list = list(alpaca)
    random.shuffle(tqa_list)
    random.shuffle(alpaca_list)

    cal_pool   = tqa_list[:600]
    eval_pool  = tqa_list[600:900]
    sft_pool   = alpaca_list  # 800 examples, cycles over N_SFT_STEPS

    # ── 2. Tokenizer (shared) ──────────────────────────────────────────────────
    print(f"\n[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 3. Base model + oracle + calibration ───────────────────────────────────
    print(f"\n[3] Loading base model for oracle + calibration...")
    base_model = load_model_eval(BASE_MODEL_ID)

    print(f"\n  Bilateral oracle (n={N_CAL})...")
    labeled_cal = bilateral_oracle(base_model, tokenizer, cal_pool, N_CAL)

    print(f"\n  Bilateral oracle — eval set (n={N_EVAL})...")
    labeled_eval = bilateral_oracle(base_model, tokenizer, eval_pool, N_EVAL)

    print(f"\n  Fisher calibration (L{SHALLOW_LAYER}→L{PROBE_LAYER})...")
    cal = calibrate(base_model, tokenizer, labeled_cal, deep=PROBE_LAYER, shallow=SHALLOW_LAYER)

    # ── 4. Condition A: BASE ───────────────────────────────────────────────────
    print(f"\n[4] Condition A — BASE (no SFT)...")
    auroc_base, jvel_base, jvel_base_std = evaluate_auroc(base_model, tokenizer, labeled_eval, cal)
    print(f"    AUROC={auroc_base:.4f} | J_vel={jvel_base:+.4f} ± {jvel_base_std:.4f} "
          f"(var={jvel_base_std**2:.5f})")

    del base_model
    torch.cuda.empty_cache()

    # ── 5. Condition B: INSTRUCT (post-RLHF) ──────────────────────────────────
    print(f"\n[5] Condition B — INSTRUCT (Δ ≈ -0.036 expected)...")
    inst_model  = load_model_eval(INSTRUCT_ID)
    auroc_inst, jvel_inst, jvel_inst_std = evaluate_auroc(inst_model, tokenizer, labeled_eval, cal)
    delta_rlhf  = auroc_inst - auroc_base
    print(f"    AUROC={auroc_inst:.4f} | J_vel={jvel_inst:+.4f} ± {jvel_inst_std:.4f} | "
          f"Δ={delta_rlhf:+.4f}")

    del inst_model
    torch.cuda.empty_cache()

    # ── 6. Condition C: Standard SFT ──────────────────────────────────────────
    print(f"\n[6] Condition C — Standard SFT ({N_SFT_STEPS} steps, CE only)...")
    model_c  = load_model_with_lora(BASE_MODEL_ID, tokenizer)
    history_c = sft_loop(model_c, tokenizer, sft_pool, labeled_eval, cal,
                          use_jvel=False, label="STD-SFT")
    auroc_std  = history_c[-1]["auroc"]
    delta_std  = auroc_std - auroc_base

    del model_c
    torch.cuda.empty_cache()

    # ── 7. Condition D: SFT + JVelocityLoss ───────────────────────────────────
    print(f"\n[7] Condition D — SFT + JVelocityLoss (α={ALPHA_JVEL})...")
    model_d   = load_model_with_lora(BASE_MODEL_ID, tokenizer)
    history_d = sft_loop(model_d, tokenizer, sft_pool, labeled_eval, cal,
                          use_jvel=True, label="JVEL-SFT")
    auroc_jvel = history_d[-1]["auroc"]
    delta_jvel = auroc_jvel - auroc_base

    del model_d
    torch.cuda.empty_cache()

    # ── 8. Verdict ─────────────────────────────────────────────────────────────
    delta_preserved  = auroc_jvel - auroc_std
    jvel_preserves   = delta_preserved > 0.0
    rlhf_reversible  = abs(delta_jvel) < abs(delta_rlhf) and delta_preserved > 0.0

    if rlhf_reversible:
        verdict = "RLHF_REVERSIBLE"
    elif jvel_preserves:
        verdict = "JVEL_PRESERVES"
    else:
        verdict = "NULL"

    # Attenuation comparison: how much does JVel recover from RLHF harm?
    rlhf_harm      = abs(delta_rlhf)           # AUROC lost by RLHF
    jvel_recovery  = max(0.0, delta_preserved)  # AUROC recovered by JVel vs std SFT
    recovery_pct   = 100.0 * jvel_recovery / max(rlhf_harm, 1e-6)

    print("\n" + "=" * 64)
    print("RESULTS — RLHF Attenuation and JVelocityLoss Reversal")
    print("=" * 64)
    print(f"  A) BASE      AUROC: {auroc_base:.4f}  Δ=  0.000  (reference)")
    print(f"  B) INSTRUCT  AUROC: {auroc_inst:.4f}  Δ={delta_rlhf:+.4f}  (RLHF attenuation)")
    print(f"  C) STD-SFT   AUROC: {auroc_std:.4f}  Δ={delta_std:+.4f}  ({N_SFT_STEPS} steps CE only)")
    print(f"  D) JVEL-SFT  AUROC: {auroc_jvel:.4f}  Δ={delta_jvel:+.4f}  ({N_SFT_STEPS} steps CE+JVel)")
    print(f"\n  JVel preservation (D-C):         {delta_preserved:+.4f}")
    print(f"  RLHF harm magnitude |Δ(B)|:      {rlhf_harm:.4f}")
    print(f"  JVel recovery of RLHF harm:      {recovery_pct:.1f}%")
    print(f"  Target |Δ(D)| < |Δ(B)|:          {'PASSED ✓' if rlhf_reversible else 'FAILED ✗'}")
    print(f"  JVel slows degradation vs STD:   {'PASSED ✓' if jvel_preserves else 'FAILED ✗'}")
    print(f"\n  VERDICT: {verdict}")
    if verdict == "RLHF_REVERSIBLE":
        print(f"  → RLHF attenuation of epistemic geometry is reversible via JVelocityLoss.")
        print(f"  → Epistemic transparency is a trainable property, not just emergent.")
    elif verdict == "JVEL_PRESERVES":
        print(f"  → JVelocityLoss slows geometry degradation during SFT.")
        print(f"  → Full RLHF reversal not achieved at {N_SFT_STEPS} steps / α={ALPHA_JVEL}.")
    print("=" * 64)

    results = {
        "meta": {
            "base_model": BASE_MODEL_ID,
            "instruct_model": INSTRUCT_ID,
            "probe_layer_deep": PROBE_LAYER,
            "probe_layer_shallow": SHALLOW_LAYER,
            "n_cal": N_CAL,
            "n_eval": N_EVAL,
            "n_sft_steps": N_SFT_STEPS,
            "alpha_jvel": ALPHA_JVEL,
            "lora_r": LORA_R,
            "cal_auroc": float(cal.auroc_cal),
        },
        "base":        {"auroc": float(auroc_base), "jvel": float(jvel_base), "jvel_std": float(jvel_base_std), "delta": 0.0},
        "instruct":    {"auroc": float(auroc_inst),  "jvel": float(jvel_inst), "jvel_std": float(jvel_inst_std), "delta": float(delta_rlhf)},
        "standard_sft":{"auroc": float(auroc_std),   "delta": float(delta_std), "history": history_c},
        "jvel_sft":    {"auroc": float(auroc_jvel),  "delta": float(delta_jvel), "history": history_d},
        "delta_preserved":   float(delta_preserved),
        "rlhf_harm":         float(rlhf_harm),
        "jvel_recovery_pct": float(recovery_pct),
        "rlhf_reversible":   rlhf_reversible,
        "jvel_preserves":    jvel_preserves,
        "verdict":           verdict,
        "interpretation": (
            "RLHF reduces epistemic transparency by degrading Fisher AUROC (Δ={:.4f}). "
            "JVelocityLoss {} this effect: preservation gain={:.4f}, "
            "recovery of RLHF harm={:.1f}%. "
            "Epistemic transparency is {} a trainable property.".format(
                delta_rlhf,
                "partially reverses" if jvel_preserves else "does not significantly mitigate",
                delta_preserved,
                recovery_pct,
                "confirmed to be" if rlhf_reversible else "not yet confirmed as",
            )
        ),
    }

    out_json = OUT_DIR / "jvelocity_training_v1_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: AUROC curves across SFT steps
        ax = axes[0]
        steps_c = [h["step"] for h in history_c]
        aucs_c  = [h["auroc"] for h in history_c]
        steps_d = [h["step"] for h in history_d]
        aucs_d  = [h["auroc"] for h in history_d]

        ax.axhline(auroc_base, color="black",  ls="--", lw=1.5, alpha=0.7, label=f"BASE {auroc_base:.3f}")
        ax.axhline(auroc_inst, color="red",    ls=":",  lw=1.5, alpha=0.7, label=f"RLHF {auroc_inst:.3f}")
        ax.plot(steps_c, aucs_c, "o-", color="orange", lw=2, label=f"Std SFT (final {auroc_std:.3f})")
        ax.plot(steps_d, aucs_d, "s-", color="green",  lw=2, label=f"JVel SFT (final {auroc_jvel:.3f})")
        ax.set_xlabel("SFT Steps")
        ax.set_ylabel("Fisher AUROC")
        ax.set_title(f"J_velocity Loss — AUROC Over Training\n{verdict}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([max(0.45, min(aucs_c + aucs_d + [auroc_inst]) - 0.05),
                     min(1.0,  max(aucs_c + aucs_d + [auroc_base]) + 0.05)])

        # Right: Final Δ bar chart
        ax2 = axes[1]
        conditions = ["BASE\n(ref)", "INSTRUCT\n(RLHF)", f"STD-SFT\n({N_SFT_STEPS}steps)", f"JVEL-SFT\n({N_SFT_STEPS}steps)"]
        deltas     = [0.0, float(delta_rlhf), float(delta_std), float(delta_jvel)]
        colors     = ["gray", "red", "orange", "green"]
        bars = ax2.bar(conditions, deltas, color=colors, alpha=0.7, edgecolor="black")
        ax2.axhline(0, color="black", lw=1)
        for bar, val in zip(bars, deltas):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + (0.001 if val >= 0 else -0.004),
                     f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
        ax2.set_ylabel("Δ AUROC vs Base")
        ax2.set_title("Final AUROC Change per Condition")
        ax2.grid(True, axis="y", alpha=0.3)

        plt.suptitle(
            f"JVelocityLoss Experiment — Qwen2.5-1.5B\n"
            f"Target: |Δ(JVel)| < |Δ(RLHF)|  →  {'PASSED ✓' if rlhf_reversible else 'FAILED ✗'}",
            fontsize=12,
        )
        plt.tight_layout()

        out_fig = OUT_DIR / "jvelocity_training_v1.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        print(f"Figure → {out_fig}")
        plt.show()

    except Exception as e:
        print(f"Plot skipped: {e}")

    return results


if __name__ == "__main__":
    run()

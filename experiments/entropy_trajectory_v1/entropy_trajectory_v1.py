#!/usr/bin/env python3
"""
entropy_trajectory_v1.py — EXP-D: Entropy Trajectory Science

SCIENTIFIC QUESTION:
  Does the entropy trajectory across generation steps 0–14 carry a
  distinguishable signature for confabulated answers vs correct answers,
  among CONFIDENT items where step-1 entropy is matched between classes?

CONTEXT:
  EXP-A tests whether Fisher+PCA64 at step-1 can detect confabulation
  when entropy is matched. EXP-D asks whether the TRAJECTORY of entropy
  across subsequent generation steps carries the signal that step-1 lacks.
  This is the "confabulation has a temporal signature" experiment.

DESIGN:
  1. Collect items in the "confident zone" (entropy < θ_conf, 30th pct).
     Three classes:
       CONFIDENT_CORRECT: entropy_step1 < θ AND F1 ≥ 0.50  (N_TARGET)
       CONFIDENT_WRONG:   entropy_step1 < θ AND F1 ≤ 0.05  (N_TARGET)
       CTX_DEP:           standard bilateral oracle CTX_DEP  (N_TARGET, baseline)

  2. For each item, generate N_STEPS tokens and record:
       entropy[t]  = Shannon entropy of logits at step t
       margin[t]   = p_top1[t] - p_top2[t]

  3. Analysis:
     a. Mean trajectory per class (plot summary in JSON)
     b. Per-step AUROC: AUROC(CC vs CW) at each step t → "AUROC vs step" curve
     c. Trajectory AUROC: LR on [H0, H1, ..., H_{N_STEPS-1}] vector
     d. Trajectory features AUROC: LR on [slope, AUC, min, max, var]
     e. Dip-then-rise test: is min(H_early) < mean(H_late) for CW but not CC?
     f. Any-step > 0.65: first step where AUROC(CC vs CW) exceeds threshold

DECISION GATE:
  TRAJECTORY_SIGNAL    : trajectory AUROC ≥ 0.65
  LATE_STEP_SIGNAL     : per-step AUROC ≥ 0.65 at any step t ≥ 3
  FEATURES_SIGNAL      : trajectory features AUROC ≥ 0.65
  TRAJECTORY_BLIND     : all ≤ 0.55 throughout
  DIP_RISE_SIGNATURE   : dip-then-rise pattern detected in CW items

MODEL: Qwen/Qwen2.5-1.5B-Instruct (Llama optional if time permits)
GPU: T4. Expected ~2-3h.
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
POOL_SIZE       = 15_000
N_TARGET        = 100                # per class
ENTROPY_PCT     = 30                 # confident zone threshold
N_STEPS         = 15                 # trajectory length (steps 0..N_STEPS-1)
LAYER_IDX       = 26                 # for bilateral oracle HS (Phase 1 only)
PARAM_MIN_F1    = 0.50
CTX_MAX_NC      = 0.05
CTX_MIN_CTX     = 0.50
TRAIN_FRAC      = 0.75
SEED            = 42
DIP_RISE_SPLIT  = 5                  # steps 0..DIP_RISE_SPLIT-1 = "early", rest = "late"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── Model ─────────────────────────────────────────────────────────────────────
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


def setup_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_hf = _get_hf_token()
    if tok_hf:
        from huggingface_hub import login
        login(token=tok_hf, add_to_git_credential=False)
    print(f"\nLoading {MODEL_ID} …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=None, trust_remote_code=True
    ).to(DEVICE).eval()
    print(f"  n_layers={mdl.config.num_hidden_layers}", flush=True)
    return mdl, tok


# ── Data ───────────────────────────────────────────────────────────────────────
def load_pool(n: int):
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      trust_remote_code=True)
    items = []
    for ex in ds:
        ep = ex.get("entity_pages", {})
        ctx = (ep.get("wiki_context") or [""])[0][:800] if ep else ""
        ans = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        items.append({"question": ex["question"], "context": ctx, "answers": ans})
        if len(items) >= n:
            break
    random.shuffle(items)
    return items


def token_f1(pred: str, golds: list) -> float:
    pt = set(pred.lower().split())
    best = 0.0
    for g in golds:
        gt = set(g.lower().split())
        if not pt or not gt:
            continue
        c = pt & gt
        if not c:
            continue
        p = len(c) / len(pt)
        r = len(c) / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


def fmt_prompt(q: str) -> str:
    return f"Answer in one short phrase.\nQuestion: {q}\nAnswer:"


def gen_text(model, tok, prompt: str, max_new: int = 60) -> str:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── Trajectory extraction ──────────────────────────────────────────────────────
def extract_trajectory(model, tok, question: str) -> list | None:
    """
    Extract entropy at steps 0..N_STEPS-1 via KV-cache decoding.
    Returns list of length N_STEPS, or None if failed.
    """
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    entropies = []

    try:
        with torch.no_grad():
            # Prefill
            pre = model(inp["input_ids"], use_cache=True)
            pkv = pre.past_key_values
            cur_ids = inp["input_ids"]

            for step in range(N_STEPS):
                out = model(
                    cur_ids[:, -1:],
                    past_key_values=pkv,
                    use_cache=True,
                )
                logits = out.logits[0, -1, :].float()
                probs  = torch.softmax(logits, dim=-1).clamp(min=1e-10)
                ent    = float(-torch.sum(probs * torch.log(probs)).item())
                entropies.append(ent)

                # Greedy decode to advance
                next_tok = logits.argmax().unsqueeze(0).unsqueeze(0)
                cur_ids  = next_tok
                pkv      = out.past_key_values

                # Stop at EOS
                if int(next_tok[0, 0].item()) == tok.eos_token_id:
                    break

    except Exception as e:
        print(f"    trajectory failed: {e}", flush=True)
        return None

    # Pad to N_STEPS if EOS hit early
    while len(entropies) < N_STEPS:
        entropies.append(entropies[-1] if entropies else 0.0)

    return entropies[:N_STEPS]


# ── Collect all classes ────────────────────────────────────────────────────────
def collect(model, tok, pool: list) -> dict:
    print("\n=== Collection Phase ===", flush=True)
    t0 = time.time()

    all_recs = []   # {entropy_step0, f1, trajectory, ctx_dep}

    for i, item in enumerate(pool):
        if (len([r for r in all_recs if r["label"] == "CC"]) >= N_TARGET and
                len([r for r in all_recs if r["label"] == "CW"]) >= N_TARGET):
            break

        if i % 100 == 0:
            cc = sum(1 for r in all_recs if r["label"] == "CC")
            cw = sum(1 for r in all_recs if r["label"] == "CW")
            print(f"  [{i}/{len(pool)}] CC={cc} CW={cw}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

        # Get trajectory (includes step-0 entropy)
        traj = extract_trajectory(model, tok, item["question"])
        if traj is None:
            continue

        entropy_step0 = traj[0]

        # Get correctness
        pred = gen_text(model, tok, fmt_prompt(item["question"]))
        f1   = token_f1(pred, item["answers"])

        all_recs.append({
            "trajectory":    traj,
            "entropy_step0": entropy_step0,
            "f1":            f1,
        })

    print(f"  Total records: {len(all_recs)}", flush=True)

    # Determine θ_conf
    all_ent0 = [r["entropy_step0"] for r in all_recs]
    theta_conf = float(np.percentile(all_ent0, ENTROPY_PCT))
    print(f"  θ_conf ({ENTROPY_PCT}th pct): {theta_conf:.4f}", flush=True)

    # Label and filter
    cc_recs, cw_recs = [], []
    for r in all_recs:
        if r["entropy_step0"] < theta_conf:
            if r["f1"] >= PARAM_MIN_F1:
                r["label"] = "CC"
                cc_recs.append(r)
            elif r["f1"] <= CTX_MAX_NC:
                r["label"] = "CW"
                cw_recs.append(r)

    random.shuffle(cc_recs); random.shuffle(cw_recs)
    cc_recs = cc_recs[:N_TARGET]
    cw_recs = cw_recs[:N_TARGET]

    print(f"  CONFIDENT_CORRECT: {len(cc_recs)}", flush=True)
    print(f"  CONFIDENT_WRONG:   {len(cw_recs)}", flush=True)

    return {
        "cc_recs":    cc_recs,
        "cw_recs":    cw_recs,
        "theta_conf": theta_conf,
    }


# ── Analysis ───────────────────────────────────────────────────────────────────
def analyze(data: dict) -> dict:
    print("\n=== Analysis ===", flush=True)

    cc_trajs = np.array([r["trajectory"] for r in data["cc_recs"]])  # (N, N_STEPS)
    cw_trajs = np.array([r["trajectory"] for r in data["cw_recs"]])

    n_cc, n_cw = len(cc_trajs), len(cw_trajs)
    if n_cc < 10 or n_cw < 10:
        return {"error": "insufficient samples", "n_cc": n_cc, "n_cw": n_cw}

    # Mean trajectory per class
    cc_mean = cc_trajs.mean(axis=0).tolist()
    cw_mean = cw_trajs.mean(axis=0).tolist()
    print(f"  CC mean trajectory (steps 0-4): "
          f"{[round(x, 3) for x in cc_mean[:5]]}", flush=True)
    print(f"  CW mean trajectory (steps 0-4): "
          f"{[round(x, 3) for x in cw_mean[:5]]}", flush=True)

    # Per-step AUROC
    y = np.array([1]*n_cc + [0]*n_cw)
    X = np.vstack([cc_trajs, cw_trajs])  # (N, N_STEPS)

    per_step_auroc = []
    for t in range(N_STEPS):
        scores = X[:, t]
        try:
            a = float(roc_auc_score(y, scores))
        except Exception:
            a = 0.5
        per_step_auroc.append(round(a, 4))
    print(f"  Per-step AUROC (steps 0-{N_STEPS-1}): {per_step_auroc}", flush=True)

    max_per_step = max(per_step_auroc)
    best_step    = per_step_auroc.index(max_per_step)
    late_max     = max(per_step_auroc[3:]) if N_STEPS > 3 else max_per_step

    # Trajectory AUROC: LR on full trajectory vector
    n_tr_cc = int(n_cc * TRAIN_FRAC)
    n_tr_cw = int(n_cw * TRAIN_FRAC)
    X_tr = np.vstack([cc_trajs[:n_tr_cc], cw_trajs[:n_tr_cw]])
    y_tr = np.array([1]*n_tr_cc + [0]*n_tr_cw)
    X_te = np.vstack([cc_trajs[n_tr_cc:], cw_trajs[n_tr_cw:]])
    y_te = np.array([1]*(n_cc-n_tr_cc) + [0]*(n_cw-n_tr_cw))

    traj_auroc = None
    try:
        lr = LogisticRegression(max_iter=500, C=0.1)
        lr.fit(X_tr, y_tr)
        sc = lr.predict_proba(X_te)[:, 1]
        traj_auroc = round(float(roc_auc_score(y_te, sc)), 4)
    except Exception as e:
        print(f"  Trajectory LR failed: {e}", flush=True)
    print(f"  Trajectory AUROC (LR on full vector): {traj_auroc}", flush=True)

    # Trajectory features AUROC
    def traj_features(trajs):
        feats = []
        for t in trajs:
            t = np.array(t)
            feats.append([
                t[1] - t[0],                        # slope (step0→step1)
                t[-1] - t[0],                       # long-range slope
                float(np.trapz(t)),                 # AUC under trajectory
                float(t.min()),                     # min entropy
                float(t.max()),                     # max entropy
                float(t.std()),                     # variance
                float(t[:DIP_RISE_SPLIT].min()),    # early min
                float(t[DIP_RISE_SPLIT:].mean()),   # late mean
            ])
        return np.array(feats, dtype=np.float32)

    feat_cc_tr = traj_features([r["trajectory"] for r in data["cc_recs"][:n_tr_cc]])
    feat_cw_tr = traj_features([r["trajectory"] for r in data["cw_recs"][:n_tr_cw]])
    feat_cc_te = traj_features([r["trajectory"] for r in data["cc_recs"][n_tr_cc:]])
    feat_cw_te = traj_features([r["trajectory"] for r in data["cw_recs"][n_tr_cw:]])

    feat_tr = np.vstack([feat_cc_tr, feat_cw_tr])
    feat_te = np.vstack([feat_cc_te, feat_cw_te])

    feat_auroc = None
    try:
        lr_f = LogisticRegression(max_iter=500, C=1.0)
        lr_f.fit(feat_tr, y_tr)
        sc_f = lr_f.predict_proba(feat_te)[:, 1]
        feat_auroc = round(float(roc_auc_score(y_te, sc_f)), 4)
    except Exception as e:
        print(f"  Feature LR failed: {e}", flush=True)
    print(f"  Trajectory features AUROC: {feat_auroc}", flush=True)

    # Dip-then-rise signature
    cc_early_min  = np.array([r["trajectory"][:DIP_RISE_SPLIT] for r in data["cc_recs"]]).min(axis=1).mean()
    cc_late_mean  = np.array([r["trajectory"][DIP_RISE_SPLIT:] for r in data["cc_recs"]]).mean(axis=1).mean()
    cw_early_min  = np.array([r["trajectory"][:DIP_RISE_SPLIT] for r in data["cw_recs"]]).min(axis=1).mean()
    cw_late_mean  = np.array([r["trajectory"][DIP_RISE_SPLIT:] for r in data["cw_recs"]]).mean(axis=1).mean()

    dip_rise_cc = cc_early_min < cc_late_mean
    dip_rise_cw = cw_early_min < cw_late_mean
    dip_rise_differential = dip_rise_cw and not dip_rise_cc

    print(f"  Dip-then-rise CC: early_min={cc_early_min:.3f} late_mean={cc_late_mean:.3f}"
          f" → {'YES' if dip_rise_cc else 'NO'}", flush=True)
    print(f"  Dip-then-rise CW: early_min={cw_early_min:.3f} late_mean={cw_late_mean:.3f}"
          f" → {'YES' if dip_rise_cw else 'NO'}", flush=True)

    # Verdict
    def verdict():
        if traj_auroc is not None and traj_auroc >= 0.65:
            return "TRAJECTORY_SIGNAL"
        if late_max >= 0.65:
            return "LATE_STEP_SIGNAL"
        if feat_auroc is not None and feat_auroc >= 0.65:
            return "FEATURES_SIGNAL"
        if dip_rise_differential:
            return "DIP_RISE_SIGNATURE"
        return "TRAJECTORY_BLIND"

    verd = verdict()
    print(f"\n  VERDICT: {verd}", flush=True)

    return {
        "n_cc": n_cc, "n_cw": n_cw,
        "n_train_per_class": min(n_tr_cc, n_tr_cw),
        "n_test_per_class":  min(n_cc-n_tr_cc, n_cw-n_tr_cw),
        "cc_mean_trajectory": [round(x, 4) for x in cc_mean],
        "cw_mean_trajectory": [round(x, 4) for x in cw_mean],
        "per_step_auroc":     per_step_auroc,
        "max_per_step_auroc": max_per_step,
        "best_step":          best_step,
        "late_step_max_auroc": round(late_max, 4),
        "trajectory_auroc":   traj_auroc,
        "features_auroc":     feat_auroc,
        "dip_rise_cc":        bool(dip_rise_cc),
        "dip_rise_cw":        bool(dip_rise_cw),
        "dip_rise_differential": bool(dip_rise_differential),
        "verdict":            verd,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"EXP-D: Entropy Trajectory v1", flush=True)
    print(f"{'='*60}", flush=True)

    model, tok = setup_model()
    pool = load_pool(POOL_SIZE)

    data = collect(model, tok, pool)
    del pool; gc.collect()

    if len(data["cc_recs"]) < 20 or len(data["cw_recs"]) < 20:
        print("FATAL: insufficient samples", flush=True)
        sys.exit(1)

    results_analysis = analyze(data)

    results = {
        "experiment": "EXP_D_ENTROPY_TRAJECTORY_V1",
        "model_name": "qwen25_1.5b_instruct",
        "config": {
            "pool_size":     POOL_SIZE,
            "n_target":      N_TARGET,
            "entropy_pct":   ENTROPY_PCT,
            "n_steps":       N_STEPS,
            "train_frac":    TRAIN_FRAC,
            "seed":          SEED,
        },
        "theta_conf": data["theta_conf"],
        **results_analysis,
    }

    out_path = "entropy_trajectory_v1_results.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items()
                   if k not in ("cc_recs", "cw_recs")}, f, indent=2)

    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

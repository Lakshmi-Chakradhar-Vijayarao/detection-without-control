#!/usr/bin/env python3
"""
entropy_trajectory_v2.py — EXP-D: Entropy Trajectory Science (fixed)

Fix vs v1:
  v1 bug: early-exit check used r["label"] before labels existed (KeyError).
  v2 fix: two-pass design.
    Pass 1 (fast scan): single generate() call per item → step-0 entropy + F1.
             ~1.5s/item × ~8000 items ≈ 3-4h. Determines theta_conf, selects CC/CW.
    Pass 2 (trajectory): extract_trajectory() ONLY for selected CC+CW items (~200).
             ~3s/item × 200 ≈ 10 min.
  Total: well within 9-hour T4 budget.

SCIENTIFIC QUESTION:
  Does entropy trajectory across steps 0–14 carry a temporal confabulation
  signature for CONFIDENT items where step-0 entropy is matched?

DESIGN (unchanged from v1):
  - Confident zone: step-0 entropy < θ_conf (30th percentile)
  - CONFIDENT_CORRECT (CC): entropy < θ AND nc_F1 >= 0.50
  - CONFIDENT_WRONG   (CW): entropy < θ AND nc_F1 <= 0.05
  - N_TARGET = 100 per class
  - Trajectory: 15 steps via KV-cache decode (no output_hidden_states)
  - Analysis: per-step AUROC curve, trajectory LR, features LR, dip-rise test

VERDICT GATE (unchanged):
  TRAJECTORY_SIGNAL   : trajectory_auroc >= 0.65
  LATE_STEP_SIGNAL    : per_step_auroc at any t>=3 >= 0.65
  FEATURES_SIGNAL     : features_auroc >= 0.65
  DIP_RISE_SIGNATURE  : dip-then-rise in CW but not CC
  TRAJECTORY_BLIND    : all <= 0.55 throughout
"""

from __future__ import annotations
import gc, json, os, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2.5-1.5B-Instruct"
POOL_SIZE      = 15_000
N_TARGET       = 100
ENTROPY_PCT    = 30
N_STEPS        = 15
PARAM_MIN_F1   = 0.50
CTX_MAX_NC     = 0.05
TRAIN_FRAC     = 0.75
SEED           = 42
DIP_RISE_SPLIT = 5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


# ── HF Token ──────────────────────────────────────────────────────────────────
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


# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Pass 1: fast scan — entropy + F1 in a single generate() call ───────────
def fast_scan(model, tok, pool: list) -> list:
    """
    Single generate() per item with output_scores=True.
    Returns list of dicts: {question, answers, entropy_step0, f1}.
    ~1.5s/item — much faster than trajectory extraction.
    """
    print(f"\n=== Pass 1: Fast Scan ({len(pool)} items) ===", flush=True)
    t0 = time.time()
    recs = []

    for i, item in enumerate(pool):
        if i % 200 == 0:
            print(f"  [{i}/{len(pool)}]  elapsed={time.time()-t0:.0f}s", flush=True)

        prompt = fmt_prompt(item["question"])
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=60,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            # Step-0 entropy from first generated token's distribution
            step0_logits = out.scores[0][0].float()
            probs = torch.softmax(step0_logits, dim=-1).clamp(min=1e-10)
            entropy = float(-torch.sum(probs * torch.log(probs)).item())

            pred = tok.decode(
                out.sequences[0][inp["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            f1 = token_f1(pred, item["answers"])

            recs.append({
                "question":    item["question"],
                "answers":     item["answers"],
                "entropy":     entropy,
                "f1":          f1,
            })
        except Exception as e:
            print(f"    item {i} failed: {e}", flush=True)

    print(f"  Pass 1 complete: {len(recs)} records  elapsed={time.time()-t0:.0f}s",
          flush=True)
    return recs


# ── Select CC and CW from fast-scan records ────────────────────────────────────
def select_candidates(recs: list) -> tuple[list, list, float]:
    """Returns (cc_candidates, cw_candidates, theta_conf)."""
    all_ent = [r["entropy"] for r in recs]
    theta_conf = float(np.percentile(all_ent, ENTROPY_PCT))
    print(f"  θ_conf ({ENTROPY_PCT}th pct): {theta_conf:.4f}", flush=True)

    cc, cw = [], []
    for r in recs:
        if r["entropy"] < theta_conf:
            if r["f1"] >= PARAM_MIN_F1:
                cc.append(r)
            elif r["f1"] <= CTX_MAX_NC:
                cw.append(r)

    random.shuffle(cc); random.shuffle(cw)
    print(f"  Confident zone candidates: CC={len(cc)}  CW={len(cw)}", flush=True)
    return cc[:N_TARGET], cw[:N_TARGET], theta_conf


# ── Pass 2: extract full trajectory for selected items ─────────────────────────
def extract_trajectory(model, tok, question: str) -> list | None:
    """
    KV-cache decode for N_STEPS steps, recording entropy at each step.
    No output_hidden_states — fast per-step decode.
    """
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    entropies = []

    try:
        with torch.no_grad():
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

                next_tok = logits.argmax().unsqueeze(0).unsqueeze(0)
                cur_ids  = next_tok
                pkv      = out.past_key_values

                if int(next_tok[0, 0].item()) == tok.eos_token_id:
                    break

    except Exception as e:
        print(f"    trajectory failed: {e}", flush=True)
        return None

    while len(entropies) < N_STEPS:
        entropies.append(entropies[-1] if entropies else 0.0)
    return entropies[:N_STEPS]


def extract_trajectories_for(model, tok, candidates: list, label: str) -> list:
    """Run trajectory extraction for a list of candidate records."""
    print(f"\n  Extracting trajectories for {label} (n={len(candidates)}) …", flush=True)
    t0 = time.time()
    results = []
    for i, cand in enumerate(candidates):
        traj = extract_trajectory(model, tok, cand["question"])
        if traj is not None:
            results.append({
                "trajectory": traj,
                "entropy":    cand["entropy"],
                "f1":         cand["f1"],
                "label":      label,
            })
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(candidates)}]  ok={len(results)}"
                  f"  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"  Done: {len(results)} trajectories  elapsed={time.time()-t0:.0f}s",
          flush=True)
    return results


# ── Analysis ───────────────────────────────────────────────────────────────────
def analyze(cc_recs: list, cw_recs: list) -> dict:
    print("\n=== Analysis ===", flush=True)

    cc_trajs = np.array([r["trajectory"] for r in cc_recs])
    cw_trajs = np.array([r["trajectory"] for r in cw_recs])
    n_cc, n_cw = len(cc_trajs), len(cw_trajs)

    if n_cc < 10 or n_cw < 10:
        return {"error": "insufficient samples", "n_cc": n_cc, "n_cw": n_cw}

    # Mean trajectory per class
    cc_mean = cc_trajs.mean(axis=0).tolist()
    cw_mean = cw_trajs.mean(axis=0).tolist()
    print(f"  CC mean (steps 0-4): {[round(x,3) for x in cc_mean[:5]]}", flush=True)
    print(f"  CW mean (steps 0-4): {[round(x,3) for x in cw_mean[:5]]}", flush=True)

    # Per-step AUROC
    y = np.array([1]*n_cc + [0]*n_cw)
    X = np.vstack([cc_trajs, cw_trajs])

    per_step_auroc = []
    for t in range(N_STEPS):
        try:
            a = float(roc_auc_score(y, X[:, t]))
        except Exception:
            a = 0.5
        per_step_auroc.append(round(a, 4))
    print(f"  Per-step AUROC: {per_step_auroc}", flush=True)

    max_per_step = max(per_step_auroc)
    best_step    = per_step_auroc.index(max_per_step)
    late_max     = max(per_step_auroc[3:]) if N_STEPS > 3 else max_per_step

    # Train/test split
    n_tr_cc = int(n_cc * TRAIN_FRAC)
    n_tr_cw = int(n_cw * TRAIN_FRAC)
    X_tr = np.vstack([cc_trajs[:n_tr_cc], cw_trajs[:n_tr_cw]])
    y_tr = np.array([1]*n_tr_cc + [0]*n_tr_cw)
    X_te = np.vstack([cc_trajs[n_tr_cc:], cw_trajs[n_tr_cw:]])
    y_te = np.array([1]*(n_cc-n_tr_cc) + [0]*(n_cw-n_tr_cw))

    # Trajectory AUROC (LR on full vector)
    traj_auroc = None
    try:
        lr = LogisticRegression(max_iter=500, C=0.1)
        lr.fit(X_tr, y_tr)
        traj_auroc = round(float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])), 4)
    except Exception as e:
        print(f"  Trajectory LR failed: {e}", flush=True)
    print(f"  Trajectory AUROC (LR): {traj_auroc}", flush=True)

    # Features AUROC
    def traj_features(trajs):
        feats = []
        for t in trajs:
            t = np.array(t)
            feats.append([
                t[1] - t[0],
                t[-1] - t[0],
                float(np.trapz(t)),
                float(t.min()),
                float(t.max()),
                float(t.std()),
                float(t[:DIP_RISE_SPLIT].min()),
                float(t[DIP_RISE_SPLIT:].mean()),
            ])
        return np.array(feats, dtype=np.float32)

    feat_tr = np.vstack([traj_features(cc_trajs[:n_tr_cc]),
                         traj_features(cw_trajs[:n_tr_cw])])
    feat_te = np.vstack([traj_features(cc_trajs[n_tr_cc:]),
                         traj_features(cw_trajs[n_tr_cw:])])
    feat_auroc = None
    try:
        lr_f = LogisticRegression(max_iter=500, C=1.0)
        lr_f.fit(feat_tr, y_tr)
        feat_auroc = round(float(roc_auc_score(y_te, lr_f.predict_proba(feat_te)[:, 1])), 4)
    except Exception as e:
        print(f"  Features LR failed: {e}", flush=True)
    print(f"  Features AUROC: {feat_auroc}", flush=True)

    # Dip-then-rise
    cc_early = np.array([r["trajectory"][:DIP_RISE_SPLIT] for r in cc_recs]).min(axis=1).mean()
    cc_late  = np.array([r["trajectory"][DIP_RISE_SPLIT:] for r in cc_recs]).mean(axis=1).mean()
    cw_early = np.array([r["trajectory"][:DIP_RISE_SPLIT] for r in cw_recs]).min(axis=1).mean()
    cw_late  = np.array([r["trajectory"][DIP_RISE_SPLIT:] for r in cw_recs]).mean(axis=1).mean()
    dip_cc   = cc_early < cc_late
    dip_cw   = cw_early < cw_late
    dip_diff = dip_cw and not dip_cc
    print(f"  Dip-rise CC: early_min={cc_early:.3f} late_mean={cc_late:.3f} → {'YES' if dip_cc else 'NO'}",
          flush=True)
    print(f"  Dip-rise CW: early_min={cw_early:.3f} late_mean={cw_late:.3f} → {'YES' if dip_cw else 'NO'}",
          flush=True)

    # Verdict
    def verdict():
        if traj_auroc is not None and traj_auroc >= 0.65:
            return "TRAJECTORY_SIGNAL"
        if late_max >= 0.65:
            return "LATE_STEP_SIGNAL"
        if feat_auroc is not None and feat_auroc >= 0.65:
            return "FEATURES_SIGNAL"
        if dip_diff:
            return "DIP_RISE_SIGNATURE"
        return "TRAJECTORY_BLIND"

    verd = verdict()
    print(f"\n  VERDICT: {verd}", flush=True)

    return {
        "n_cc": n_cc, "n_cw": n_cw,
        "n_train_per_class": min(n_tr_cc, n_tr_cw),
        "n_test_per_class":  min(n_cc - n_tr_cc, n_cw - n_tr_cw),
        "cc_mean_trajectory": [round(x, 4) for x in cc_mean],
        "cw_mean_trajectory": [round(x, 4) for x in cw_mean],
        "per_step_auroc":     per_step_auroc,
        "max_per_step_auroc": max_per_step,
        "best_step":          best_step,
        "late_step_max_auroc": round(late_max, 4),
        "trajectory_auroc":   traj_auroc,
        "features_auroc":     feat_auroc,
        "dip_rise_cc":        bool(dip_cc),
        "dip_rise_cw":        bool(dip_cw),
        "dip_rise_differential": bool(dip_diff),
        "cc_early_min":       round(float(cc_early), 4),
        "cc_late_mean":       round(float(cc_late), 4),
        "cw_early_min":       round(float(cw_early), 4),
        "cw_late_mean":       round(float(cw_late), 4),
        "verdict":            verd,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}", flush=True)
    print(f"EXP-D: Entropy Trajectory v2 (two-pass)", flush=True)
    print(f"{'='*60}", flush=True)

    model, tok = setup_model()
    pool = load_pool(POOL_SIZE)
    print(f"Pool: {len(pool)} items", flush=True)

    # Pass 1: fast scan
    scan_recs = fast_scan(model, tok, pool)
    del pool; gc.collect()

    if len(scan_recs) < 200:
        print("FATAL: too few scan records", flush=True)
        sys.exit(1)

    # Select CC / CW candidates
    cc_cands, cw_cands, theta_conf = select_candidates(scan_recs)

    if len(cc_cands) < 20 or len(cw_cands) < 20:
        print(f"FATAL: insufficient candidates CC={len(cc_cands)} CW={len(cw_cands)}", flush=True)
        sys.exit(1)

    cc_entropy_mean = float(np.mean([r["entropy"] for r in cc_cands]))
    cw_entropy_mean = float(np.mean([r["entropy"] for r in cw_cands]))
    print(f"  CC entropy mean={cc_entropy_mean:.3f}  CW entropy mean={cw_entropy_mean:.3f}",
          flush=True)

    # Pass 2: trajectory extraction for selected items only
    cc_recs = extract_trajectories_for(model, tok, cc_cands, "CC")
    cw_recs = extract_trajectories_for(model, tok, cw_cands, "CW")

    if len(cc_recs) < 10 or len(cw_recs) < 10:
        print("FATAL: trajectory extraction yielded too few items", flush=True)
        sys.exit(1)

    results_analysis = analyze(cc_recs, cw_recs)

    results = {
        "experiment":    "EXP_D_ENTROPY_TRAJECTORY_V2",
        "model_name":    "qwen25_1.5b_instruct",
        "config": {
            "pool_size":     len(scan_recs),
            "n_target":      N_TARGET,
            "entropy_pct":   ENTROPY_PCT,
            "n_steps":       N_STEPS,
            "train_frac":    TRAIN_FRAC,
            "seed":          SEED,
        },
        "theta_conf":          theta_conf,
        "cc_entropy_mean":     cc_entropy_mean,
        "cw_entropy_mean":     cw_entropy_mean,
        **results_analysis,
    }

    out_path = "entropy_trajectory_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

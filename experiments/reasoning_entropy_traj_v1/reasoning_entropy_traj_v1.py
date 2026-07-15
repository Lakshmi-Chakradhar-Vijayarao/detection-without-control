"""
EXP-G: Reasoning Model Answer Entropy Trajectory  (v3)
=======================================================
After </think>, does the answer entropy trajectory of DeepSeek-R1-Distill-Qwen-1.5B
show the same burst pattern as base Qwen (EXP-D)?

Scientific question: Does the think block pre-resolve epistemic uncertainty?
  Prediction: FLAT trajectory (low entropy uniformly) — think block pre-resolved
  Alternative: BURST at step ~4 — same as base model; think block doesn't help

If FLAT: reasoning pre-resolves epistemic state. Burst is a base model artifact.
If BURST: the burst occurs regardless of reasoning. Epistemic uncertainty persists into answer.

This connects EXP-D (base model trajectory) to EXP-B/C (reasoning models, C019, C022).

v3 changes (dataset switch):
  - TriviaQA → GSM8K (math word problems). DeepSeek-R1-Distill was trained on math;
    TriviaQA factual recall is near-zero for this model (CC=0 after 40 items in v2).
  - Answer scoring: numeric exact match on value after #### delimiter.
  - POOL_SIZE reduced (GSM8K test=1319 items, ~40-50% expected accuracy).
  - MAX_THINK increased 512 → 1024 (math reasoning needs more thinking tokens).
"""

import gc, json, os, random, re, time
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
THINK_END_STR = "</think>"

POOL_SIZE    = 1319     # full GSM8K test split
N_CC_TARGET  = 50
N_CW_TARGET  = 50
N_STEPS      = 15       # answer trajectory steps
MAX_THINK    = 1024     # math needs more thinking tokens than trivia
MAX_ANS      = 100
SEED         = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

rng = random.Random(SEED)
np.random.seed(SEED)


# ── Utilities ────────────────────────────────────────────────────────────────
def _get_hf_token():
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(k, "")
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def extract_numeric(text: str) -> str | None:
    """Extract final numeric answer: prefer #### format, fallback to last number."""
    if "####" in text:
        candidate = text.split("####")[-1].strip().replace(",", "").split()[0]
        candidate = re.sub(r"[^\d\.\-]", "", candidate)
        if candidate:
            return candidate
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def is_correct(pred: str, gold: str) -> bool:
    pred_num = extract_numeric(pred)
    if pred_num is None:
        return False
    try:
        return abs(float(pred_num) - float(gold)) < 1e-3
    except (ValueError, TypeError):
        return pred_num.strip() == gold.strip()


def fmt_prompt(q: str) -> str:
    return f"<|im_start|>user\n{q}\nPlease reason step by step and put your final answer after ####.<|im_end|>\n<|im_start|>assistant\n<think>\n"


def load_pool(n: int):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    pool = []
    for ex in ds:
        if len(pool) >= n:
            break
        q = ex["question"]
        ans_str = ex["answer"]
        if "####" in ans_str:
            gold = ans_str.split("####")[-1].strip().replace(",", "")
            gold = re.sub(r"[^\d\.\-]", "", gold)
            if q and gold:
                pool.append({"q": q, "gold": gold})
    rng.shuffle(pool)
    return pool


def output_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1).clamp(min=1e-10)
    return float(-torch.sum(probs * torch.log(probs)).item())


# ── Reasoning generation + answer entropy extraction ─────────────────────────
def generate_and_extract_answer_traj(mdl, tok, question: str) -> dict | None:
    """Generate full reasoning chain, then extract per-step entropy of ANSWER tokens."""
    prompt = fmt_prompt(question)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    think_end_id = tok.encode(THINK_END_STR, add_special_tokens=False)
    think_end_id = think_end_id[-1] if think_end_id else None  # last token of </think>

    # Phase 1: generate think block
    with torch.no_grad():
        pre = mdl(inp["input_ids"], use_cache=True)
        pkv = pre.past_key_values

    cur_ids = inp["input_ids"]
    answer_started = False
    answer_tokens  = []
    answer_entropies = []
    think_entropies  = []

    for step in range(MAX_THINK + MAX_ANS):
        with torch.no_grad():
            out = mdl(cur_ids[:, -1:], past_key_values=pkv, use_cache=True)
        logits = out.logits[0, -1, :]
        ent = output_entropy(logits)
        next_id = int(logits.argmax().item())
        pkv = out.past_key_values
        cur_ids = torch.tensor([[next_id]], device=DEVICE)

        if not answer_started:
            think_entropies.append(ent)
            if think_end_id is not None and next_id == think_end_id:
                answer_started = True
        else:
            answer_entropies.append(ent)
            answer_tokens.append(next_id)
            if len(answer_entropies) >= N_STEPS:
                break
            if next_id == tok.eos_token_id:
                break

    if not answer_started or len(answer_entropies) < 3:
        return None

    # Pad to N_STEPS
    while len(answer_entropies) < N_STEPS:
        answer_entropies.append(answer_entropies[-1])
    answer_entropies = answer_entropies[:N_STEPS]

    answer_text = tok.decode(answer_tokens, skip_special_tokens=True).strip()
    return {
        "answer": answer_text,
        "answer_traj": answer_entropies,
        "think_len": len(think_entropies),
        "step0_ent": think_entropies[0] if think_entropies else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"EXP-G v3: Reasoning Model Answer Entropy Trajectory (GSM8K)", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"N_target: {N_CC_TARGET} CC, {N_CW_TARGET} CW  N_steps: {N_STEPS}", flush=True)

    hf_token = _get_hf_token()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True, token=hf_token
    ).to(DEVICE).eval()

    pool = load_pool(POOL_SIZE)

    cc_trajs, cw_trajs = [], []
    t0 = time.time()

    for i, item in enumerate(pool):
        if len(cc_trajs) >= N_CC_TARGET and len(cw_trajs) >= N_CW_TARGET:
            break

        q, gold = item["q"], item["gold"]

        result = generate_and_extract_answer_traj(mdl, tok, q)
        if result is None:
            continue

        correct = is_correct(result["answer"], gold)

        collected_before = len(cc_trajs) + len(cw_trajs)
        if correct and len(cc_trajs) < N_CC_TARGET:
            cc_trajs.append(result["answer_traj"])
        elif not correct and len(cw_trajs) < N_CW_TARGET:
            cw_trajs.append(result["answer_traj"])
        collected_after = len(cc_trajs) + len(cw_trajs)

        # Intermediate save every 20 collected items
        if collected_after > collected_before and collected_after % 20 == 0:
            interim = {
                "experiment": "EXP_G_REASONING_ENTROPY_TRAJ_V1",
                "model": MODEL_ID,
                "status": "in_progress",
                "n_cc": len(cc_trajs),
                "n_cw": len(cw_trajs),
                "cc_trajs": [[round(x, 4) for x in t] for t in cc_trajs],
                "cw_trajs": [[round(x, 4) for x in t] for t in cw_trajs],
            }
            with open("/kaggle/working/exp_g_interim.json", "w") as f:
                json.dump(interim, f)
            print(f"  [interim save] CC={len(cc_trajs)} CW={len(cw_trajs)}", flush=True)

        if (i+1) % 20 == 0:
            print(f"  [{i+1}] CC={len(cc_trajs)} CW={len(cw_trajs)}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    n_min = min(len(cc_trajs), len(cw_trajs))
    print(f"\nCollected: n_min={n_min}/class", flush=True)

    if n_min < 20:
        print("INSUFFICIENT — aborting", flush=True)
        return

    cc_arr = np.array(cc_trajs[:n_min])   # (n, N_STEPS)
    cw_arr = np.array(cw_trajs[:n_min])

    cc_mean = cc_arr.mean(axis=0).tolist()
    cw_mean = cw_arr.mean(axis=0).tolist()

    # Per-step AUROC
    per_step_auroc = []
    for s in range(N_STEPS):
        y = np.array([1] * n_min + [0] * n_min)
        scores = np.concatenate([cc_arr[:, s], cw_arr[:, s]])
        try:
            a = float(roc_auc_score(y, scores))
        except Exception:
            a = 0.5
        per_step_auroc.append(round(a, 4))

    # Trajectory AUROC (flatten all steps)
    X = np.hstack([cc_arr, cw_arr]).T    # (2n, N_STEPS)
    X = np.vstack([cc_arr, cw_arr])
    y_all = np.array([1] * n_min + [0] * n_min)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, C=0.1)
    lr.fit(X, y_all)
    traj_auroc = float(roc_auc_score(y_all, lr.decision_function(X)))

    # Classify pattern
    max_step = int(np.argmax(per_step_auroc))
    peak_auroc = per_step_auroc[max_step]
    step0_auroc = per_step_auroc[0]
    pattern = "BURST" if (peak_auroc - step0_auroc > 0.10 and max_step >= 2) else "FLAT"

    print(f"\n{'='*60}", flush=True)
    print(f"TRAJECTORY AUROC: {traj_auroc:.4f}", flush=True)
    print(f"PER-STEP AUROC: {per_step_auroc}", flush=True)
    print(f"PATTERN: {pattern}  (peak at step {max_step} = {peak_auroc:.4f})", flush=True)
    print(f"CC MEAN ENTROPY TRAJ: {[round(x,3) for x in cc_mean]}", flush=True)
    print(f"CW MEAN ENTROPY TRAJ: {[round(x,3) for x in cw_mean]}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {
        "experiment": "EXP_G_REASONING_ENTROPY_TRAJ_V1",
        "model": MODEL_ID,
        "n_per_class": n_min,
        "n_steps": N_STEPS,
        "trajectory_auroc": round(traj_auroc, 4),
        "per_step_auroc": per_step_auroc,
        "cc_mean_traj": [round(x, 4) for x in cc_mean],
        "cw_mean_traj": [round(x, 4) for x in cw_mean],
        "pattern": pattern,
        "peak_step": max_step,
        "status": "complete"
    }
    with open("/kaggle/working/exp_g_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: /kaggle/working/exp_g_results.json", flush=True)


if __name__ == "__main__":
    main()

"""
Continuous Oracle — Excluded Zone Signal Test
===============================================

PRE-REGISTERED PREDICTION (written 2026-07-13, before any results seen):
  Signal exists in the bilateral oracle excluded zone (nocontext F1 ∈ 0.10-0.40).
  Fisher+PCA64 probe separates soft-PARAM from soft-CTX_DEP in this ambiguous region.
  AUROC_middle >= 0.65 at N >= 50 per class.

  Kill criterion: AUROC_middle < 0.50 (chance performance).

Background:
  Standard bilateral oracle drops items with nocontext F1 ∈ (0.05, 0.50) — the
  "excluded zone." This creates a potential selection-bias concern: maybe the clean
  signal in PARAM/CTX_DEP extremes is an artifact of cherry-picking easy cases.
  This experiment tests whether signal *also* exists in the ambiguous middle.

Protocol:
  SOFT_PARAM  = nocontext F1 ∈ [0.20, 0.50) — partial knowledge
  SOFT_CTXDEP = nocontext F1 ∈ [0.05, 0.20) AND withcontext F1 >= 0.50
                — context helps items the model partially knows
  Model: Qwen2.5-1.5B-Instruct, Layer 26, step-1
  Dataset: TriviaQA validation (same as primary experiments)
"""

from __future__ import annotations
import subprocess
print("[init] installing bitsandbytes...", flush=True)
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=False)

import os, json, time, string

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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER = 26
N_TARGET    = 100    # per class (excluded zone has fewer items)
MAX_SCAN    = 10000
N_PCA       = 64
SEED        = 42

# Standard oracle thresholds
F1_PARAM_HARD = 0.50   # clean PARAM floor
F1_CTX_CEIL   = 0.05   # clean CTX_DEP nocontext ceiling

# Excluded zone thresholds
F1_SOFT_PARAM_LO  = 0.20   # soft-PARAM: nocontext F1 in [0.20, 0.50)
F1_SOFT_PARAM_HI  = 0.50
F1_SOFT_CTX_LO    = 0.05   # soft-CTX_DEP: nocontext F1 in [0.05, 0.20)
F1_SOFT_CTX_HI    = 0.20
F1_WC_MIN         = 0.50   # withcontext floor for soft-CTX_DEP

MAX_NEW_TOKENS = 60
MAX_CTX_LEN    = 800
N_BOOTSTRAP    = 1000
OUTPUT_FILE    = "/kaggle/working/continuous_oracle_v1.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
T0 = time.time()
def ts(): return f"[{int(time.time()-T0):5d}s]"

print(f"{ts()} === CONTINUOUS ORACLE v1 ===")
print(f"{ts()} PRE-REGISTERED: AUROC_middle >= 0.65 in excluded zone F1 ∈ [0.05, 0.50)")
print(f"{ts()} Kill criterion: AUROC_middle < 0.50")
print(f"{ts()} Model={MODEL_ID}  Layer={PROBE_LAYER}  N_TARGET={N_TARGET}")
print(f"{ts()} Device={DEVICE}")

# ── F1 helpers ────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = sum(min(p_toks.count(t), g_toks.count(t)) for t in set(p_toks) & set(g_toks))
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    return 2 * prec * rec / (prec + rec)

# ── Model ─────────────────────────────────────────────────────────────────────
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

print(f"{ts()} Loading tokenizer...")
_hf_tok = _get_hf_token()
if _hf_tok:
    from huggingface_hub import login
    login(token=_hf_tok, add_to_git_credential=False)
    print(f"{ts()} HF login OK")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=_hf_tok, trust_remote_code=True)
print(f"{ts()} Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=None,
    token=_hf_tok,
    trust_remote_code=True,
)
model = model.to(DEVICE)
model.eval()
print(f"{ts()} Model loaded. layers={model.config.num_hidden_layers}  d={model.config.hidden_size}")

# ── Dataset ───────────────────────────────────────────────────────────────────
print(f"{ts()} Loading TriviaQA validation set...")
ds = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=True)  # rc has search_results; rc.nocontext strips context → wc pass never fires
print(f"{ts()} TriviaQA validation: {len(ds)} items")

def format_nocontext(question: str) -> str:
    return (f"<|im_start|>user\nAnswer the following question with a short phrase "
            f"(1-5 words). If you don't know, say 'I don't know'.\n\n"
            f"Question: {question}\n<|im_end|>\n<|im_start|>assistant\n")

def format_withcontext(question: str, context: str) -> str:
    ctx_trunc = context[:MAX_CTX_LEN]
    return (f"<|im_start|>user\nUsing the following passage, answer the question "
            f"with a short phrase (1-5 words).\n\n"
            f"Passage: {ctx_trunc}\n\nQuestion: {question}\n<|im_end|>\n"
            f"<|im_start|>assistant\n")

def generate_and_extract(prompt: str, collect_hs: bool = False):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = enc.input_ids.shape[1]

    hs_vec = None
    if collect_hs:
        hook_storage = {}
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # With KV cache each decode step has shape[1]==1; prefill has shape[1]==input_len.
            # Capture only once at step-1 (first new token).
            if "hs" not in hook_storage and hs.shape[1] == 1:
                hook_storage["hs"] = hs[:, -1, :].detach().float().cpu()
        handle = model.model.layers[PROBE_LAYER].register_forward_hook(hook)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            output_hidden_states=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if collect_hs:
        handle.remove()
        hs_vec = hook_storage.get("hs", None)
        if hs_vec is not None:
            hs_vec = hs_vec.squeeze(0).numpy()

    new_tokens = out[0, input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, hs_vec

def get_answers(item) -> list[str]:
    ans = item.get("answer", {})
    aliases = ans.get("aliases", []) if isinstance(ans, dict) else []
    norm_val = ans.get("normalized_value", "") if isinstance(ans, dict) else str(ans)
    val = ans.get("value", "") if isinstance(ans, dict) else str(ans)
    return list({a for a in [norm_val, val] + aliases if a})

def get_context(item) -> str:
    # rc split: search_results["description"] and entity_pages["wiki_context"] are lists of strings
    sr = item.get("search_results", {})
    if isinstance(sr, dict):
        for key in ("description", "search_context"):
            val = sr.get(key, [])
            if isinstance(val, list) and val and val[0]:
                return val[0]
            if isinstance(val, str) and val:
                return val
    ep = item.get("entity_pages", {})
    if isinstance(ep, dict):
        wc = ep.get("wiki_context", [])
        if isinstance(wc, list) and wc and wc[0]:
            return wc[0]
    return ""

# ── Scan ──────────────────────────────────────────────────────────────────────
soft_param_items  = []  # nocontext F1 ∈ [0.20, 0.50)
soft_ctxdep_items = []  # nocontext F1 ∈ [0.05, 0.20), withcontext F1 >= 0.50

n_scanned = 0
n_soft_param = 0
n_soft_ctxdep = 0

# Also collect standard oracle items for reference comparison
hard_param_items  = []
hard_ctxdep_items = []
n_hard_param = 0
n_hard_ctxdep = 0
HARD_TARGET = 150

print(f"{ts()} Starting soft-oracle scan...")
print(f"{ts()} SOFT_PARAM: nocontext F1 ∈ [{F1_SOFT_PARAM_LO}, {F1_SOFT_PARAM_HI})")
print(f"{ts()} SOFT_CTX_DEP: nocontext F1 ∈ [{F1_SOFT_CTX_LO}, {F1_SOFT_CTX_HI}), wc F1 >= {F1_WC_MIN}")

for item in ds:
    if (n_soft_param >= N_TARGET and n_soft_ctxdep >= N_TARGET
            and n_hard_param >= HARD_TARGET and n_hard_ctxdep >= HARD_TARGET):
        break
    if n_scanned >= MAX_SCAN:
        break

    question = item["question"].strip()
    answers  = get_answers(item)
    if not answers:
        continue

    n_scanned += 1
    nc_prompt = format_nocontext(question)
    nc_answer, nc_hs = generate_and_extract(nc_prompt, collect_hs=True)
    nc_f1 = token_f1(nc_answer, answers[0]) if len(answers) == 1 else max(token_f1(nc_answer, a) for a in answers)

    # Hard PARAM
    if nc_f1 >= F1_PARAM_HARD and n_hard_param < HARD_TARGET:
        hard_param_items.append({"hs": nc_hs, "nc_f1": nc_f1})
        n_hard_param += 1

    # Soft PARAM: excluded zone upper half
    elif F1_SOFT_PARAM_LO <= nc_f1 < F1_SOFT_PARAM_HI and n_soft_param < N_TARGET:
        soft_param_items.append({"hs": nc_hs, "nc_f1": nc_f1, "question": question})
        n_soft_param += 1
        if n_soft_param % 10 == 0:
            print(f"{ts()} SOFT_PARAM={n_soft_param}/{N_TARGET} SOFT_CTX={n_soft_ctxdep}/{N_TARGET} scanned={n_scanned}")

    # Soft CTX_DEP: excluded zone lower half — needs withcontext pass
    elif F1_SOFT_CTX_LO <= nc_f1 < F1_SOFT_CTX_HI and n_soft_ctxdep < N_TARGET:
        context = get_context(item)
        if context:
            wc_prompt = format_withcontext(question, context)
            wc_answer, _ = generate_and_extract(wc_prompt, collect_hs=False)
            wc_f1 = max(token_f1(wc_answer, a) for a in answers)
            if wc_f1 >= F1_WC_MIN:
                soft_ctxdep_items.append({"hs": nc_hs, "nc_f1": nc_f1, "wc_f1": wc_f1, "question": question})
                n_soft_ctxdep += 1
                if n_soft_ctxdep % 10 == 0:
                    print(f"{ts()} SOFT_PARAM={n_soft_param}/{N_TARGET} SOFT_CTX={n_soft_ctxdep}/{N_TARGET} scanned={n_scanned}")

    # Hard CTX_DEP
    elif nc_f1 <= F1_CTX_CEIL and n_hard_ctxdep < HARD_TARGET:
        context = get_context(item)
        if context:
            wc_prompt = format_withcontext(question, context)
            wc_answer, _ = generate_and_extract(wc_prompt, collect_hs=False)
            wc_f1 = max(token_f1(wc_answer, a) for a in answers)
            if wc_f1 >= 0.50:
                hard_ctxdep_items.append({"hs": nc_hs, "nc_f1": nc_f1, "wc_f1": wc_f1})
                n_hard_ctxdep += 1

    if n_scanned % 100 == 0:
        print(f"{ts()} scan {n_scanned}/{MAX_SCAN}  sp={n_soft_param}  sc={n_soft_ctxdep}  hp={n_hard_param}  hc={n_hard_ctxdep}")

print(f"{ts()} Scan done: {n_scanned} scanned")
print(f"{ts()} Soft: PARAM={n_soft_param}, CTX_DEP={n_soft_ctxdep}")
print(f"{ts()} Hard: PARAM={n_hard_param}, CTX_DEP={n_hard_ctxdep}")

# ── Probe ─────────────────────────────────────────────────────────────────────
def run_probe(pos_items, neg_items, label: str):
    n = min(len(pos_items), len(neg_items))
    if n < 30:
        return {"label": label, "status": "INSUFFICIENT", "n": n}

    X = np.vstack([
        np.array([x["hs"] for x in pos_items[:n]]),
        np.array([x["hs"] for x in neg_items[:n]])
    ])
    y = np.array([1]*n + [0]*n)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.35, random_state=SEED, stratify=y
    )
    n_pca = min(N_PCA, X_tr.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_tr_pca, y_tr)
    scores = lda.decision_function(X_te_pca)
    auroc = roc_auc_score(y_te, scores)

    y_shuf = y_te.copy(); np.random.shuffle(y_shuf)
    auroc_shuf = roc_auc_score(y_shuf, scores)

    rng = np.random.default_rng(SEED)
    bs_vals = [roc_auc_score(y_te[idx := rng.choice(len(y_te), len(y_te))], scores[idx])
               for _ in range(N_BOOTSTRAP)]
    return {
        "label": label,
        "n_per_class": n,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "auroc": float(auroc),
        "auroc_shuffled": float(auroc_shuf),
        "shuffled_clean": bool(auroc_shuf < auroc - 0.05),
        "bootstrap_ci_lo": float(np.percentile(bs_vals, 2.5)),
        "bootstrap_ci_hi": float(np.percentile(bs_vals, 97.5)),
    }

print(f"{ts()} Running probes...")

soft_result = run_probe(soft_param_items, soft_ctxdep_items, "SOFT_ORACLE")
hard_result = run_probe(hard_param_items, hard_ctxdep_items, "HARD_ORACLE")

print(f"{ts()} === RESULTS ===")
for r in [soft_result, hard_result]:
    if r.get("status") == "INSUFFICIENT":
        print(f"{ts()} {r['label']}: INSUFFICIENT N={r['n']}")
    else:
        print(f"{ts()} {r['label']}: AUROC={r['auroc']:.4f}  shuf={r['auroc_shuffled']:.4f}  N={r['n_per_class']}/class")

# Verdict
if soft_result.get("status") == "INSUFFICIENT":
    verdict = "PROTOCOL_FAILURE"
elif soft_result["auroc"] >= 0.65 and soft_result["shuffled_clean"]:
    verdict = "SIGNAL_IN_EXCLUDED_ZONE"
elif soft_result["auroc"] >= 0.50 and soft_result["shuffled_clean"]:
    verdict = "WEAK_SIGNAL"
else:
    verdict = "NO_SIGNAL"

kill_triggered = soft_result.get("auroc", 0) < 0.50

print(f"{ts()} VERDICT = {verdict}")
print(f"{ts()} PRE-REGISTERED = AUROC_middle >= 0.65")
print(f"{ts()} Kill triggered = {kill_triggered}")

result = {
    "experiment": "EXP_CONTINUOUS_ORACLE_V1",
    "pre_registered_prediction": "AUROC_middle >= 0.65",
    "kill_criterion": "AUROC_middle < 0.50",
    "model": MODEL_ID,
    "dataset": "trivia_qa",
    "layer": PROBE_LAYER,
    "soft_oracle": soft_result,
    "hard_oracle": hard_result,
    "verdict": verdict,
    "prediction_met": soft_result.get("auroc", 0) >= 0.65,
    "kill_triggered": kill_triggered,
    "protocol_note": (
        "Soft PARAM = nocontext F1 in [0.20, 0.50). "
        "Soft CTX_DEP = nocontext F1 in [0.05, 0.20), withcontext F1 >= 0.50. "
        "Hard oracle (standard protocol) run in parallel for comparison."
    ),
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=2, cls=_NpEncoder)

print(f"{ts()} Results saved to {OUTPUT_FILE}")
print(f"{ts()} DONE — verdict={verdict}")

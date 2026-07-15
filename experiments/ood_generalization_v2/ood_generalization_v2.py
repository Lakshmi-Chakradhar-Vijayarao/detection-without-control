#!/usr/bin/env python3
"""
ood_generalization_v2.py — EXP_OOD_GENERALIZATION_V2

BUG FIX from v1:
  v1 used bare chat template (no brevity instruction). Qwen generates verbose
  answers like "Paris is the capital of France..." → token_f1 precision drag →
  F1 << 0.50 for correct answers → PARAM=1 after 200 scans → crash.
  Fix: add system prompt "Answer in one short phrase." + answer_contains fallback.

SCIENTIFIC QUESTION:

SCIENTIFIC QUESTION:
  Does the bilateral oracle Fisher+PCA64 probe trained on TriviaQA transfer
  without retraining to other knowledge tasks (MMLU, HotpotQA)?

  Motivated by C009 (task-specific geometry, cross-task cosim 0.004–0.035) and
  the EMNLP 2025 independent finding showing 25pp AUROC drop for OOD probe transfer.

  This experiment defines the scope of all observability claims: if the probe is
  task-specific, every claim in the program must be qualified as TriviaQA-specific.
  If the probe generalizes, the bilateral oracle protocol is a domain-agnostic instrument.

DESIGN:
  Phase 1 — Source probe: train bilateral oracle Fisher+PCA64 on TriviaQA
    N=200/class PARAM/CTX_DEP, Layer 26, step-1
    Record pca + lda probe weights

  Phase 2 — OOD transfer (no retraining):
    Task A: MMLU-STEM (factual, parametric-dominant)
      Collect KNOWS (model answers correctly without context, F1>=0.50) and
      DOESNT_KNOW (model answers incorrectly without context, F1<=0.05) items
      Apply TriviaQA probe → transfer AUROC
    Task B: HotpotQA (multi-hop, context-dependent)
      Bilateral oracle: no-context pass (question only) vs with-context pass
      (question + supporting_facts). PARAM/CTX_DEP by same criteria as TriviaQA.
      Apply TriviaQA probe → transfer AUROC

  Phase 3 — Within-OOD calibration (fresh probe per OOD task):
    Train fresh Fisher+PCA64 on N=100/class of OOD items
    Measure within-AUROC
    Compute gap = within_AUROC - transfer_AUROC

  Hidden states: always from the no-context pass (matching TriviaQA protocol).

VERDICTS:
  OOD_GENERALIZES: transfer_AUROC >= within_AUROC - 0.05 on BOTH tasks
  OOD_PARTIAL:     transfer_AUROC >= 0.65 on at least one task
  TASK_SPECIFIC:   transfer_AUROC < 0.65 on both tasks
                   (consistent with C009 cross-task cosim; all claims
                    require TriviaQA-specific qualification)

GPU: T4 (~8h)
Model: Qwen/Qwen2.5-1.5B-Instruct (same as bilateral oracle source)
"""

from __future__ import annotations
import gc, json, os, random, time
import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_ID         = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX        = 26
N_SOURCE         = 200    # TriviaQA per class (source probe)
N_OOD_TRANSFER   = 100    # per class for OOD transfer evaluation
N_OOD_WITHIN     = 100    # per class for within-OOD fresh probe
PCA_DIM          = 64
MAX_NEW          = 60
PARAM_MIN_F1     = 0.50
CTX_DEP_MAX_F1   = 0.05

SOURCE_POOL_SIZE  = 10_000
MMLU_POOL_SIZE    = 3_000   # MMLU-STEM subset
HPQA_POOL_SIZE    = 3_000

RESULTS_FILE     = "/kaggle/working/ood_generalization_v1_results.json"
INTERMEDIATE     = "/kaggle/working/ood_generalization_v1_intermediate.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("GPU required.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)


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

_tok_val = _get_hf_token()
if _tok_val:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_tok_val, add_to_git_credential=False)
    print("HF login: OK", flush=True)


# ── Model ────────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_ID} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded: {n_layers} layers", flush=True)
    return model, tokenizer


_BRIEF = "Answer the following question in one short phrase or name only. Do not explain."


def prompt_nc(tokenizer, q: str) -> str:
    msgs = [
        {"role": "system", "content": _BRIEF},
        {"role": "user", "content": q},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def prompt_wc(tokenizer, q: str, context: str) -> str:
    content = f"Context: {context}\n\n{q}"
    msgs = [
        {"role": "system", "content": _BRIEF},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_hs_and_entropy(model, tokenizer, prompt: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    hs_out = [None]

    def hook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        hs_out[0] = x[:, -1, :].detach().float().cpu().numpy()

    h = model.model.layers[LAYER_IDX].register_forward_hook(hook)
    with torch.no_grad():
        out = model(ids)
    h.remove()

    logits = out.logits[0, -1, :].float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
    probs  = torch.softmax(logits, dim=-1)
    ent    = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
    if not np.isfinite(ent):
        ent = 0.0
    hs = hs_out[0][0] if hs_out[0] is not None else None
    return hs, ent


def generate(model, tokenizer, prompt: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def token_f1(pred: str, golds) -> float:
    p = set(pred.lower().split())
    best = 0.0
    for g in golds:
        q = set(g.lower().split())
        c = p & q
        if c and p and q:
            pr = len(c) / len(p)
            rc = len(c) / len(q)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def answer_contains(pred: str, golds) -> bool:
    pl = pred.lower()
    return any(g.lower().strip() and g.lower().strip() in pl for g in golds)


# ── Probe ────────────────────────────────────────────────────────────────────────

def fit_probe(X, y, pca_dim=PCA_DIM):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pca = PCA(n_components=min(pca_dim, X.shape[1], len(X) - 1))
    Xp = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xp, y)
    return pca, lda


def eval_probe(pca, lda, X_test, y_test, X_ent_test=None):
    from sklearn.metrics import roc_auc_score
    Xp = pca.transform(X_test)
    scores = lda.decision_function(Xp)
    auroc = float(roc_auc_score(y_test, scores))
    # Shuffled control
    y_shuf = y_test.copy(); np.random.shuffle(y_shuf)
    auroc_shuf = float(roc_auc_score(y_shuf, scores))
    result = {"auroc": auroc, "shuffled": auroc_shuf}
    if X_ent_test is not None:
        try:
            ent_auroc = float(roc_auc_score(y_test, -X_ent_test))
            result["entropy_auroc"] = ent_auroc
        except Exception:
            pass
    return result


def bootstrap_ci(pca, lda, X_test, y_test, n_boot=500, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    Xp = pca.transform(X_test)
    scores = lda.decision_function(Xp)
    aurocs = []
    n = len(y_test)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            a = float(roc_auc_score(y_test[idx], scores[idx]))
            aurocs.append(a)
        except Exception:
            pass
    return (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5)))


# ── Data: TriviaQA ────────────────────────────────────────────────────────────────

def load_trivia_qa_pool(n=SOURCE_POOL_SIZE):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train", streaming=True)
    items = []
    for row in ds:
        if not row["answer"]["aliases"]:
            continue
        items.append({"question": row["question"], "answers": row["answer"]["aliases"]})
        if len(items) >= n:
            break
    np.random.shuffle(items)
    print(f"TriviaQA pool: {len(items)} items", flush=True)
    return items


def collect_trivia_bilateral(model, tokenizer, pool, n_target=N_SOURCE):
    param_items, ctx_dep_items = [], []
    n_scanned = 0
    for item in pool:
        if len(param_items) >= n_target and len(ctx_dep_items) >= n_target:
            break
        q = item["question"]
        golds = item["answers"]
        # No-context pass
        p_nc = prompt_nc(tokenizer, q)
        ans_nc = generate(model, tokenizer, p_nc)
        f1_nc = token_f1(ans_nc, golds)
        n_scanned += 1

        param_ok = f1_nc >= PARAM_MIN_F1 or answer_contains(ans_nc, golds)
        if param_ok and len(param_items) < n_target:
            hs, ent = get_hs_and_entropy(model, tokenizer, p_nc)
            if hs is not None:
                param_items.append({"hs": hs, "ent": ent})
        elif f1_nc <= CTX_DEP_MAX_F1 and not answer_contains(ans_nc, golds) and len(ctx_dep_items) < n_target:
            # With-context pass: gold answer provided directly to verify CTX_DEP
            context = f"The answer to this question is: {golds[0]}."
            p_wc = prompt_wc(tokenizer, q, context)
            ans_wc = generate(model, tokenizer, p_wc)
            wc_ok = token_f1(ans_wc, golds) >= PARAM_MIN_F1 or answer_contains(ans_wc, golds)
            if wc_ok:
                hs, ent = get_hs_and_entropy(model, tokenizer, p_nc)
                if hs is not None:
                    ctx_dep_items.append({"hs": hs, "ent": ent})

        if n_scanned % 200 == 0:
            print(f"  TriviaQA scanned={n_scanned} PARAM={len(param_items)} CTX_DEP={len(ctx_dep_items)}", flush=True)

    print(f"TriviaQA collected: PARAM={len(param_items)}, CTX_DEP={len(ctx_dep_items)}, scanned={n_scanned}", flush=True)
    return param_items, ctx_dep_items


# ── Data: MMLU ────────────────────────────────────────────────────────────────────

MMLU_STEM_SUBJECTS = [
    "high_school_physics", "high_school_chemistry", "high_school_biology",
    "high_school_mathematics", "high_school_computer_science",
    "college_physics", "college_chemistry", "college_biology",
    "college_mathematics", "college_computer_science",
    "elementary_mathematics", "abstract_algebra", "anatomy",
    "astronomy", "clinical_knowledge", "conceptual_physics",
    "electrical_engineering", "formal_logic", "machine_learning",
    "medical_genetics", "miscellaneous", "nutrition", "prehistory",
    "professional_medicine", "world_history", "world_religions",
]


def _mmlu_to_str(row) -> tuple[str, str]:
    choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"]))
    q = f"{row['question']}\n{choices_str}"
    answer_letter = chr(65 + row["answer"])
    answer_text = row["choices"][row["answer"]]
    return q, answer_text


def collect_mmlu_items(model, tokenizer, n_target=N_OOD_TRANSFER):
    """Collect KNOWS (correct without context) vs DOESNT_KNOW (wrong without context) from MMLU-STEM."""
    from datasets import load_dataset
    knows_items, doesnt_items = [], []
    n_scanned = 0
    # Load multiple subjects to get enough items
    for subj in MMLU_STEM_SUBJECTS:
        if len(knows_items) >= n_target and len(doesnt_items) >= n_target:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
        except Exception:
            try:
                ds = load_dataset("lukaemon/mmlu", subj, split="test", trust_remote_code=True)
            except Exception:
                continue
        items = list(ds)
        np.random.shuffle(items)
        for row in items:
            if len(knows_items) >= n_target and len(doesnt_items) >= n_target:
                break
            q_str, answer_text = _mmlu_to_str(row)
            p_nc = prompt_nc(tokenizer, q_str)
            ans_nc = generate(model, tokenizer, p_nc)
            f1_nc = token_f1(ans_nc, [answer_text])
            n_scanned += 1

            knows_ok = f1_nc >= PARAM_MIN_F1 or answer_contains(ans_nc, [answer_text])
            if knows_ok and len(knows_items) < n_target:
                hs, ent = get_hs_and_entropy(model, tokenizer, p_nc)
                if hs is not None:
                    knows_items.append({"hs": hs, "ent": ent})
            elif f1_nc <= CTX_DEP_MAX_F1 and not answer_contains(ans_nc, [answer_text]) and len(doesnt_items) < n_target:
                hs, ent = get_hs_and_entropy(model, tokenizer, p_nc)
                if hs is not None:
                    doesnt_items.append({"hs": hs, "ent": ent})

        if n_scanned % 200 == 0:
            print(f"  MMLU scanned={n_scanned} KNOWS={len(knows_items)} DOESNT={len(doesnt_items)}", flush=True)

    print(f"MMLU collected: KNOWS={len(knows_items)}, DOESNT_KNOW={len(doesnt_items)}, scanned={n_scanned}", flush=True)
    return knows_items, doesnt_items


# ── Data: HotpotQA ────────────────────────────────────────────────────────────────

def collect_hotpotqa_bilateral(model, tokenizer, n_target=N_OOD_TRANSFER):
    """Bilateral oracle on HotpotQA using provided supporting_facts as context."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
    param_items, ctx_dep_items = [], []
    n_scanned = 0
    for row in ds:
        if len(param_items) >= n_target and len(ctx_dep_items) >= n_target:
            break
        q = row["question"]
        golds = [row["answer"]]
        # Build context from supporting facts
        supporting = []
        for title, sents in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
            # Find the actual sentences from context
            for ctx_title, ctx_sents in zip(row["context"]["title"], row["context"]["sentences"]):
                if ctx_title == title and sents < len(ctx_sents):
                    supporting.append(ctx_sents[sents])
                    break
        context_str = " ".join(supporting[:3])  # up to 3 supporting sentences

        # No-context pass
        p_nc = prompt_nc(tokenizer, q)
        ans_nc = generate(model, tokenizer, p_nc)
        f1_nc = token_f1(ans_nc, golds)
        n_scanned += 1

        param_ok = f1_nc >= PARAM_MIN_F1 or answer_contains(ans_nc, golds)
        if param_ok and len(param_items) < n_target:
            hs, ent = get_hs_and_entropy(model, tokenizer, p_nc)
            if hs is not None:
                param_items.append({"hs": hs, "ent": ent})
        elif f1_nc <= CTX_DEP_MAX_F1 and not answer_contains(ans_nc, golds) and context_str and len(ctx_dep_items) < n_target:
            p_wc = prompt_wc(tokenizer, q, context_str)
            ans_wc = generate(model, tokenizer, p_wc)
            wc_ok = token_f1(ans_wc, golds) >= PARAM_MIN_F1 or answer_contains(ans_wc, golds)
            if wc_ok:
                hs, ent = get_hs_and_entropy(model, tokenizer, p_nc)
                if hs is not None:
                    ctx_dep_items.append({"hs": hs, "ent": ent})

        if n_scanned % 200 == 0:
            print(f"  HotpotQA scanned={n_scanned} PARAM={len(param_items)} CTX_DEP={len(ctx_dep_items)}", flush=True)

    print(f"HotpotQA collected: PARAM={len(param_items)}, CTX_DEP={len(ctx_dep_items)}, scanned={n_scanned}", flush=True)
    return param_items, ctx_dep_items


# ── Arrays ────────────────────────────────────────────────────────────────────────

def items_to_arrays(pos_items, neg_items):
    hs_pos = np.array([x["hs"] for x in pos_items])
    hs_neg = np.array([x["hs"] for x in neg_items])
    ent_pos = np.array([x["ent"] for x in pos_items])
    ent_neg = np.array([x["ent"] for x in neg_items])
    X = np.vstack([hs_pos, hs_neg]).astype(np.float32)
    y = np.array([1] * len(pos_items) + [0] * len(neg_items))
    ent = np.concatenate([ent_pos, ent_neg])
    return X, y, ent


def save_intermediate(data: dict):
    with open(INTERMEDIATE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Intermediate saved: {INTERMEDIATE}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    results = {}
    T0 = time.time()
    def ts(): return f"[{int(time.time()-T0)}s]"

    model, tokenizer = load_model()

    # ── Phase 1: Source probe (TriviaQA) ─────────────────────────────────────────
    print(f"\n{ts()} === Phase 1: TriviaQA source probe ===", flush=True)
    trivia_pool = load_trivia_qa_pool(SOURCE_POOL_SIZE)
    param_items, ctx_dep_items = collect_trivia_bilateral(model, tokenizer, trivia_pool, N_SOURCE)

    n_src = min(len(param_items), len(ctx_dep_items))
    if n_src < 50:
        raise RuntimeError(f"Insufficient source items: PARAM={len(param_items)}, CTX_DEP={len(ctx_dep_items)}")

    X_src, y_src, ent_src = items_to_arrays(
        param_items[:n_src], ctx_dep_items[:n_src]
    )
    # Split: 80% train, 20% test
    split = int(0.8 * len(y_src))
    idx = np.random.permutation(len(y_src))
    tr_idx, te_idx = idx[:split], idx[split:]
    pca_src, lda_src = fit_probe(X_src[tr_idx], y_src[tr_idx])
    src_result = eval_probe(pca_src, lda_src, X_src[te_idx], y_src[te_idx], ent_src[te_idx])
    src_ci = bootstrap_ci(pca_src, lda_src, X_src[te_idx], y_src[te_idx])
    src_result["ci_95"] = src_ci
    src_result["n_train"] = int(len(tr_idx) // 2)
    src_result["n_test"] = int(len(te_idx) // 2)
    print(f"{ts()} TriviaQA source AUROC={src_result['auroc']:.4f} shuffled={src_result['shuffled']:.4f}", flush=True)
    results["source_trivia"] = src_result

    save_intermediate(results)
    del trivia_pool; gc.collect()

    # ── Phase 2a: MMLU OOD transfer ───────────────────────────────────────────────
    print(f"\n{ts()} === Phase 2a: MMLU OOD transfer ===", flush=True)
    mmlu_knows, mmlu_doesnt = collect_mmlu_items(model, tokenizer, N_OOD_TRANSFER)

    n_mmlu = min(len(mmlu_knows), len(mmlu_doesnt))
    if n_mmlu >= 30:
        X_mmlu, y_mmlu, ent_mmlu = items_to_arrays(mmlu_knows[:n_mmlu], mmlu_doesnt[:n_mmlu])
        transfer_mmlu = eval_probe(pca_src, lda_src, X_mmlu, y_mmlu, ent_mmlu)
        transfer_ci_mmlu = bootstrap_ci(pca_src, lda_src, X_mmlu, y_mmlu)
        transfer_mmlu["ci_95"] = transfer_ci_mmlu
        transfer_mmlu["n"] = n_mmlu
        print(f"{ts()} MMLU transfer AUROC={transfer_mmlu['auroc']:.4f} shuffled={transfer_mmlu['shuffled']:.4f}", flush=True)

        # Within-MMLU probe (fresh)
        n_within = min(n_mmlu, N_OOD_WITHIN)
        X_w, y_w, ent_w = items_to_arrays(
            mmlu_knows[:n_within], mmlu_doesnt[:n_within]
        )
        sp = int(0.8 * len(y_w))
        idx2 = np.random.permutation(len(y_w))
        pca_w, lda_w = fit_probe(X_w[idx2[:sp]], y_w[idx2[:sp]])
        within_mmlu = eval_probe(pca_w, lda_w, X_w[idx2[sp:]], y_w[idx2[sp:]])
        print(f"{ts()} MMLU within AUROC={within_mmlu['auroc']:.4f}", flush=True)

        results["mmlu"] = {
            "transfer": transfer_mmlu,
            "within": within_mmlu,
            "gap": round(within_mmlu["auroc"] - transfer_mmlu["auroc"], 4),
            "n_items": n_mmlu,
        }
    else:
        print(f"{ts()} MMLU insufficient items: KNOWS={len(mmlu_knows)} DOESNT={len(mmlu_doesnt)}", flush=True)
        results["mmlu"] = {"error": "insufficient items", "n_knows": len(mmlu_knows), "n_doesnt": len(mmlu_doesnt)}

    save_intermediate(results)
    del mmlu_knows, mmlu_doesnt; gc.collect()

    # ── Phase 2b: HotpotQA OOD transfer ──────────────────────────────────────────
    print(f"\n{ts()} === Phase 2b: HotpotQA OOD transfer ===", flush=True)
    hpqa_param, hpqa_ctx = collect_hotpotqa_bilateral(model, tokenizer, N_OOD_TRANSFER)

    n_hpqa = min(len(hpqa_param), len(hpqa_ctx))
    if n_hpqa >= 30:
        X_hpqa, y_hpqa, ent_hpqa = items_to_arrays(hpqa_param[:n_hpqa], hpqa_ctx[:n_hpqa])
        transfer_hpqa = eval_probe(pca_src, lda_src, X_hpqa, y_hpqa, ent_hpqa)
        transfer_ci_hpqa = bootstrap_ci(pca_src, lda_src, X_hpqa, y_hpqa)
        transfer_hpqa["ci_95"] = transfer_ci_hpqa
        transfer_hpqa["n"] = n_hpqa
        print(f"{ts()} HotpotQA transfer AUROC={transfer_hpqa['auroc']:.4f} shuffled={transfer_hpqa['shuffled']:.4f}", flush=True)

        n_within = min(n_hpqa, N_OOD_WITHIN)
        X_w2, y_w2, ent_w2 = items_to_arrays(
            hpqa_param[:n_within], hpqa_ctx[:n_within]
        )
        sp2 = int(0.8 * len(y_w2))
        idx3 = np.random.permutation(len(y_w2))
        pca_w2, lda_w2 = fit_probe(X_w2[idx3[:sp2]], y_w2[idx3[:sp2]])
        within_hpqa = eval_probe(pca_w2, lda_w2, X_w2[idx3[sp2:]], y_w2[idx3[sp2:]])
        print(f"{ts()} HotpotQA within AUROC={within_hpqa['auroc']:.4f}", flush=True)

        results["hotpotqa"] = {
            "transfer": transfer_hpqa,
            "within": within_hpqa,
            "gap": round(within_hpqa["auroc"] - transfer_hpqa["auroc"], 4),
            "n_items": n_hpqa,
        }
    else:
        print(f"{ts()} HotpotQA insufficient: PARAM={len(hpqa_param)} CTX_DEP={len(hpqa_ctx)}", flush=True)
        results["hotpotqa"] = {"error": "insufficient items", "n_param": len(hpqa_param), "n_ctx_dep": len(hpqa_ctx)}

    save_intermediate(results)

    # ── Verdict ───────────────────────────────────────────────────────────────────
    verdicts = []
    for task in ["mmlu", "hotpotqa"]:
        r = results.get(task, {})
        if "transfer" in r:
            t_auroc = r["transfer"]["auroc"]
            w_auroc = r["within"]["auroc"]
            gap = r["gap"]
            if t_auroc >= w_auroc - 0.05:
                verdicts.append(f"{task.upper()}: OOD_GENERALIZES (transfer={t_auroc:.3f}, within={w_auroc:.3f}, gap={gap:+.3f})")
            elif t_auroc >= 0.65:
                verdicts.append(f"{task.upper()}: OOD_PARTIAL (transfer={t_auroc:.3f}, within={w_auroc:.3f}, gap={gap:+.3f})")
            else:
                verdicts.append(f"{task.upper()}: TASK_SPECIFIC (transfer={t_auroc:.3f} < 0.65)")
        else:
            verdicts.append(f"{task.upper()}: SKIPPED (insufficient items)")

    n_generalizes = sum(1 for v in verdicts if "OOD_GENERALIZES" in v)
    n_partial = sum(1 for v in verdicts if "OOD_PARTIAL" in v)
    n_specific = sum(1 for v in verdicts if "TASK_SPECIFIC" in v)

    if n_generalizes == 2:
        overall = "OOD_GENERALIZES"
    elif n_generalizes + n_partial >= 1:
        overall = "OOD_PARTIAL"
    else:
        overall = "TASK_SPECIFIC"

    results["verdict"] = overall
    results["verdict_detail"] = verdicts
    results["src_auroc"] = src_result["auroc"]
    results["elapsed_s"] = int(time.time() - T0)

    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL VERDICT: {overall}", flush=True)
    for v in verdicts:
        print(f"  {v}", flush=True)
    print(f"Source (TriviaQA) AUROC: {src_result['auroc']:.4f}", flush=True)
    print(f"Elapsed: {results['elapsed_s']}s", flush=True)
    print(f"{'='*60}", flush=True)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()

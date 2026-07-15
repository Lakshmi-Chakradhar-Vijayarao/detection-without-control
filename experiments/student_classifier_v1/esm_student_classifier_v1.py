"""
esm_student_classifier_v1.py — Student classifier as pluggable EpistemicStateVector backend

Architecture claim: the AdaptiveScheduler consumes EpistemicStateVector.
The extraction backend is pluggable. This experiment validates the student
backend (axis="student") which deploys WITHOUT hidden-state access.

EpistemicStateVector.axis = "fisher_lda"  → Qwen2.5-1.5B hidden states (full GPU)
EpistemicStateVector.axis = "student"     → DistilBERT text only (<1ms, CPU)
EpistemicStateVector.axis = "neural_probe"→ learned probe (future)

The scheduler receives the same interface regardless of backend. If DistilBERT
achieves AUROC ≥ 0.70, the full routing system deploys without any access to
the main model's hidden states — critical for API-based or black-box deployments.

Prior baseline (Kill Criterion 2): direct text classification → AUROC 0.53-0.57
This experiment:  L2 distillation from J_know scores → target AUROC ≥ 0.70

Three student architectures compared:
  T1) TF-IDF + Ridge regression        — text baseline (no teacher distillation)
  T2) SBERT + MLP (regression head)    — semantic baseline
  T3) DistilBERT fine-tuned on J_know  — distillation student (target backend)

The teacher is Qwen2.5-1.5B Fisher probe (calibrated on 200 TriviaQA questions).
Training data: 500 questions with J_know regression labels from teacher.
Test data:     200 held-out questions.

Deployment flow with student backend:
  state = EpistemicStateVector.from_j_know(
      j_know  = student_model.predict(query_text),   # DistilBERT, CPU, <1ms
      axis    = "student",
      layer   = -1,
  )
  decision = AdaptiveScheduler().schedule(state, seq_len)
  if decision.retrieval_needed:
      context = rag.query(query_text)
  answer = main_llm(query_text, context)

Verdict: STUDENT_VIABLE if T3 AUROC ≥ 0.70 on held-out set

Kaggle GPU: T4 16GB
Teacher: Qwen/Qwen2.5-1.5B (Fisher probe, layer 22)
Student: distilbert-base-uncased (fine-tuned) + TF-IDF + SBERT baselines
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Dependencies ───────────────────────────────────────────────────────────────

def _ensure(pkg, name=None):
    try:
        __import__(name or pkg)
    except ImportError:
        os.system(f"pip install {pkg} -q")

_ensure("datasets")
_ensure("scikit-learn", "sklearn")
_ensure("sentence-transformers", "sentence_transformers")
_ensure("matplotlib")

from datasets import load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── Constants ──────────────────────────────────────────────────────────────────

TEACHER_ID       = "Qwen/Qwen2.5-1.5B"
STUDENT_ID       = "distilbert-base-uncased"
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    _sm_major, _sm_minor = cc
    _sm = _sm_major * 10 + _sm_minor
    assert _sm >= 70, f"GPU sm_{_sm} not supported — need T4 (sm_75) or better. Re-run on T4."
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DTYPE            = torch.float16 if DEVICE == "cuda" else torch.float32

PROBE_LAYER      = 22       # ~79% depth in 28-layer 1.5B
SHALLOW_LAYER    = 8

N_CAL            = 200      # teacher calibration
N_TRAIN          = 500      # student training labels from teacher
N_TEST           = 200      # held-out test
N_STUDENT_EPOCHS = 5
STUDENT_LR       = 2e-5
STUDENT_BATCH    = 16
STUDENT_MAX_LEN  = 128

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Teacher: Fisher probe ─────────────────────────────────────────────────────

@dataclass
class FisherProbe:
    diff_u:        np.ndarray
    c_ctx:         np.ndarray
    layer_deep:    int
    layer_shallow: int
    theta:         float
    auroc_cal:     float


def get_transformer_layers(model: nn.Module):
    for path in ["base_model.model.model.layers", "model.layers", "model.model.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("Cannot locate transformer layers.")


def bilateral_oracle(model, tokenizer, questions, n) -> List[Dict]:
    """Assign PARAM/CTX_DEP labels via no-context correctness test."""
    model.eval()
    labeled = []
    for q in questions:
        if len(labeled) >= n:
            break
        question = q.get("question", "")
        ans_d = q.get("answer", {})
        gold  = [ans_d.get("value", "")] + ans_d.get("aliases", [])
        gold  = [a.lower().strip() for a in gold if a]
        if not gold:
            continue

        prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).lower().strip()
        correct = any(g in pred or pred in g for g in gold if len(g) > 2)
        labeled.append({"question": question, "gold": gold[:2], "label": "PARAM" if correct else "CTX_DEP"})

    print(f"    Oracle: PARAM={sum(1 for x in labeled if x['label']=='PARAM')}, "
          f"CTX_DEP={sum(1 for x in labeled if x['label']=='CTX_DEP')}")
    return labeled


def extract_j_know(model, tokenizer, questions, probe: FisherProbe) -> np.ndarray:
    """Extract continuous J_know scores for a list of questions."""
    model.eval()
    layers = get_transformer_layers(model)
    diff_u = torch.tensor(probe.diff_u, dtype=torch.float32, device=DEVICE)
    c_ctx  = torch.tensor(probe.c_ctx,  dtype=torch.float32, device=DEVICE)

    j_scores = []
    deep_buf: Dict = {}

    def _hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        deep_buf["h"] = h[:, -1, :].float().detach()

    handle = layers[probe.layer_deep].register_forward_hook(_hook)

    for q in questions:
        prompt = f"Answer briefly.\nQuestion: {q['question']}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).input_ids.to(DEVICE)
        with torch.no_grad():
            model(ids)
        if "h" in deep_buf:
            hd = deep_buf.pop("h")[0]
            j_scores.append(((hd - c_ctx) * diff_u).sum().item())

    handle.remove()
    return np.array(j_scores, dtype=np.float32)


def calibrate_fisher(model, tokenizer, labeled_cal) -> FisherProbe:
    """Calibrate Fisher LDA probe from bilateral oracle labels."""
    model.eval()
    layers = get_transformer_layers(model)

    # Extract hidden states at both layers
    hs_deep, hs_sha, y_list = [], [], []
    deep_buf, sha_buf = {}, {}

    def _hook_deep(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        deep_buf["h"] = h[:, -1, :].float().detach().cpu().numpy()

    def _hook_sha(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        sha_buf["h"] = h[:, -1, :].float().detach().cpu().numpy()

    h_dep = layers[PROBE_LAYER].register_forward_hook(_hook_deep)
    h_sha = layers[SHALLOW_LAYER].register_forward_hook(_hook_sha)

    for q in labeled_cal:
        prompt = f"Answer briefly.\nQuestion: {q['question']}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200).input_ids.to(DEVICE)
        with torch.no_grad():
            model(ids)
        if "h" in deep_buf and "h" in sha_buf:
            hs_deep.append(deep_buf.pop("h")[0])
            hs_sha.append(sha_buf.pop("h")[0])
            y_list.append(1 if q["label"] == "PARAM" else 0)

    h_dep.remove()
    h_sha.remove()

    X_deep = np.array(hs_deep, dtype=np.float32)
    y      = np.array(y_list, dtype=np.int32)

    lda    = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_deep, y)
    diff_u = lda.scalings_[:, 0].astype(np.float32)
    diff_u /= np.linalg.norm(diff_u) + 1e-8
    c_ctx  = X_deep[y == 0].mean(axis=0).astype(np.float32)

    j_deep  = (X_deep - c_ctx) @ diff_u
    theta   = float((j_deep[y==1].mean() + j_deep[y==0].mean()) / 2.0)
    auroc   = roc_auc_score(y, j_deep)
    print(f"    Fisher calibration AUROC: {auroc:.4f} | θ={theta:.3f}")

    return FisherProbe(diff_u=diff_u, c_ctx=c_ctx, layer_deep=PROBE_LAYER,
                       layer_shallow=SHALLOW_LAYER, theta=theta, auroc_cal=auroc)


# ── Student T1: TF-IDF + Ridge ─────────────────────────────────────────────────

def train_tfidf_ridge(
    train_questions: List[Dict], train_j: np.ndarray,
    test_questions:  List[Dict], test_y: np.ndarray,
) -> Dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    train_texts = [q["question"] for q in train_questions]
    test_texts  = [q["question"] for q in test_questions]

    # Regression on continuous J_know
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(train_texts, train_j)
    test_preds = pipe.predict(test_texts)

    auroc = roc_auc_score(test_y, test_preds)
    print(f"    T1 TF-IDF+Ridge AUROC: {auroc:.4f}")
    return {"name": "TF-IDF+Ridge", "auroc": float(auroc), "preds": test_preds.tolist()}


# ── Student T2: SBERT + MLP ────────────────────────────────────────────────────

def train_sbert_mlp(
    train_questions: List[Dict], train_j: np.ndarray,
    test_questions:  List[Dict], test_y: np.ndarray,
) -> Dict:
    from sentence_transformers import SentenceTransformer

    print("    Loading SBERT (all-MiniLM-L6-v2)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    train_texts = [q["question"] for q in train_questions]
    test_texts  = [q["question"] for q in test_questions]

    print("    Encoding train + test...")
    X_train = sbert.encode(train_texts, batch_size=64, show_progress_bar=False)
    X_test  = sbert.encode(test_texts,  batch_size=64, show_progress_bar=False)

    # 2-layer MLP regression on J_know
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(train_j,  dtype=torch.float32).to(DEVICE)
    X_te = torch.tensor(X_test,  dtype=torch.float32).to(DEVICE)

    D = X_tr.shape[1]
    mlp = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    mlp.train()
    for epoch in range(30):
        perm = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], 32):
            idx = perm[i:i+32]
            opt.zero_grad()
            pred = mlp(X_tr[idx]).squeeze(-1)
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            opt.step()

    mlp.eval()
    with torch.no_grad():
        preds = mlp(X_te).squeeze(-1).cpu().numpy()

    auroc = roc_auc_score(test_y, preds)
    print(f"    T2 SBERT+MLP AUROC: {auroc:.4f}")
    return {"name": "SBERT+MLP", "auroc": float(auroc), "preds": preds.tolist()}


# ── Student T3: DistilBERT regression ─────────────────────────────────────────

class DistilBERTRegressor(nn.Module):
    """DistilBERT with a scalar regression head predicting J_know."""

    def __init__(self, model_id: str = STUDENT_ID):
        super().__init__()
        from transformers import DistilBertModel
        self.bert   = DistilBertModel.from_pretrained(model_id)
        self.head   = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] token
        return self.head(cls).squeeze(-1)


def train_distilbert(
    train_questions: List[Dict], train_j: np.ndarray,
    test_questions:  List[Dict], test_y: np.ndarray,
) -> Dict:
    print(f"    Fine-tuning {STUDENT_ID} on {len(train_questions)} J_know labels...")

    student_tok = AutoTokenizer.from_pretrained(STUDENT_ID)

    def encode(questions, max_len=STUDENT_MAX_LEN):
        texts = [q["question"] for q in questions]
        enc   = student_tok(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_len)
        return enc.input_ids, enc.attention_mask

    # Build train / test tensors
    tr_ids, tr_mask = encode(train_questions)
    te_ids, te_mask = encode(test_questions)
    tr_j = torch.tensor(train_j, dtype=torch.float32)

    model = DistilBERTRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STUDENT_LR, weight_decay=0.01)
    total_steps = (len(train_questions) // STUDENT_BATCH) * N_STUDENT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    loss_fn = nn.MSELoss()

    n = len(train_questions)
    for epoch in range(N_STUDENT_EPOCHS):
        model.train()
        perm  = torch.randperm(n)
        total_loss = 0.0
        steps = 0
        for i in range(0, n, STUDENT_BATCH):
            idx = perm[i:i + STUDENT_BATCH]
            ids  = tr_ids[idx].to(DEVICE)
            mask = tr_mask[idx].to(DEVICE)
            tgt  = tr_j[idx].to(DEVICE)

            optimizer.zero_grad()
            pred = model(ids, mask)
            loss = loss_fn(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"      epoch {epoch+1}/{N_STUDENT_EPOCHS} — MSE loss: {avg_loss:.4f}")

    # Evaluate
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_questions), STUDENT_BATCH):
            ids  = te_ids[i:i+STUDENT_BATCH].to(DEVICE)
            mask = te_mask[i:i+STUDENT_BATCH].to(DEVICE)
            out  = model(ids, mask)
            all_preds.extend(out.cpu().numpy().tolist())

    preds = np.array(all_preds)
    auroc = roc_auc_score(test_y, preds)
    print(f"    T3 DistilBERT AUROC: {auroc:.4f}")

    # Save student model for potential deployment
    model_path = OUT_DIR / "student_distilbert.pt"
    torch.save(model.state_dict(), model_path)
    print(f"    Student weights → {model_path}")

    return {"name": "DistilBERT-student", "auroc": float(auroc), "preds": preds.tolist()}


# ── Routing simulation ─────────────────────────────────────────────────────────

def simulate_routing(
    preds:     np.ndarray,
    test_y:    np.ndarray,
    threshold: float,
    label:     str,
) -> Dict:
    """
    Simulate pre-inference routing:
      pred ≥ threshold → PARAM → skip RAG
      pred < threshold → CTX_DEP → add RAG context

    Reports: RAG skip rate, precision/recall of PARAM routing.
    """
    routing = (preds >= threshold).astype(int)

    n_param_pred   = routing.sum()
    rag_skip_rate  = float(n_param_pred / len(routing))

    # Among predicted PARAM, how many were actually PARAM? (precision)
    tp = int(((routing == 1) & (test_y == 1)).sum())
    fp = int(((routing == 1) & (test_y == 0)).sum())
    fn = int(((routing == 0) & (test_y == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)

    return {
        "student":        label,
        "threshold":      float(threshold),
        "rag_skip_rate":  round(rag_skip_rate, 3),
        "param_precision": round(precision, 3),
        "param_recall":   round(recall, 3),
        "correctly_skipped": tp,
        "incorrectly_skipped": fp,
    }


# ── Main experiment ────────────────────────────────────────────────────────────

def run():
    print("=" * 64)
    print("STUDENT CLASSIFIER EXPERIMENT v1")
    print(f"Teacher: {TEACHER_ID} | Student: {STUDENT_ID}")
    print("=" * 64)

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n[1] Loading TriviaQA...")
    tqa = load_dataset("trivia_qa", "rc.nocontext", split="train[:2500]", trust_remote_code=True)
    tqa_list = list(tqa)
    random.shuffle(tqa_list)

    cal_pool   = tqa_list[:400]
    train_pool = tqa_list[400:1200]   # oracle labels for student training
    test_pool  = tqa_list[1200:1600]  # held-out

    # ── 2. Teacher model + calibration ────────────────────────────────────────
    print(f"\n[2] Loading teacher: {TEACHER_ID}...")
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token
    teacher_tok.padding_side = "left"

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, torch_dtype=DTYPE, device_map=None, trust_remote_code=True,
    ).to(DEVICE).eval()

    print(f"\n  Bilateral oracle — calibration (n={N_CAL})...")
    labeled_cal = bilateral_oracle(teacher, teacher_tok, cal_pool, N_CAL)

    print(f"\n  Fisher calibration (L{SHALLOW_LAYER}→L{PROBE_LAYER})...")
    probe = calibrate_fisher(teacher, teacher_tok, labeled_cal)

    # Calibration quality diagnostic
    param_n  = sum(1 for x in labeled_cal if x["label"] == "PARAM")
    ctxdep_n = sum(1 for x in labeled_cal if x["label"] == "CTX_DEP")
    balance  = min(param_n, ctxdep_n) / max(param_n, ctxdep_n, 1)
    print(f"  Calibration quality: AUROC={probe.auroc_cal:.4f} | "
          f"PARAM={param_n} CTX_DEP={ctxdep_n} | "
          f"class_balance={balance:.2f} | "
          f"{'GOOD' if probe.auroc_cal >= 0.70 and balance >= 0.5 else 'WARN: low AUROC or imbalanced'}")

    # ── 3. Generate student training labels ───────────────────────────────────
    print(f"\n[3] Oracle — train set (n={N_TRAIN})...")
    labeled_train = bilateral_oracle(teacher, teacher_tok, train_pool, N_TRAIN)

    print(f"\n  Extracting J_know scores for train set...")
    train_j = extract_j_know(teacher, teacher_tok, labeled_train, probe)
    train_y = np.array([1 if q["label"] == "PARAM" else 0 for q in labeled_train], dtype=np.int32)

    # ── 4. Generate test labels ────────────────────────────────────────────────
    print(f"\n[4] Oracle — test set (n={N_TEST})...")
    labeled_test = bilateral_oracle(teacher, teacher_tok, test_pool, N_TEST)

    print(f"\n  Extracting J_know scores for test set (teacher reference)...")
    test_j = extract_j_know(teacher, teacher_tok, labeled_test, probe)
    test_y = np.array([1 if q["label"] == "PARAM" else 0 for q in labeled_test], dtype=np.int32)

    # Teacher AUROC on test set (upper bound for students)
    teacher_auroc = roc_auc_score(test_y, test_j)
    print(f"\n  Teacher AUROC on test set: {teacher_auroc:.4f} (upper bound)")

    del teacher
    torch.cuda.empty_cache()

    # ── 5. Train students ──────────────────────────────────────────────────────
    results_students = []

    print(f"\n[5] T1: TF-IDF + Ridge (text baseline)...")
    r_t1 = train_tfidf_ridge(labeled_train, train_j, labeled_test, test_y)
    results_students.append(r_t1)

    print(f"\n[6] T2: SBERT + MLP (semantic baseline)...")
    r_t2 = train_sbert_mlp(labeled_train, train_j, labeled_test, test_y)
    results_students.append(r_t2)

    print(f"\n[7] T3: DistilBERT regression (distillation student)...")
    r_t3 = train_distilbert(labeled_train, train_j, labeled_test, test_y)
    results_students.append(r_t3)

    # ── 6. Routing simulation ──────────────────────────────────────────────────
    print(f"\n[8] Routing simulation (threshold = probe θ={probe.theta:.3f})...")

    routing_results = []
    for r in results_students:
        preds = np.array(r["preds"])
        # Normalize preds to approx J_know range for threshold comparison
        scaler = StandardScaler()
        preds_n = scaler.fit_transform(preds.reshape(-1,1)).ravel() * float(test_j.std()) + float(test_j.mean())
        sim = simulate_routing(preds_n, test_y, threshold=probe.theta, label=r["name"])
        routing_results.append(sim)
        print(f"    {r['name']:20s}: RAG skip={sim['rag_skip_rate']:.1%} | "
              f"precision={sim['param_precision']:.3f} | recall={sim['param_recall']:.3f}")

    # Teacher routing (oracle upper bound)
    teacher_sim = simulate_routing(test_j, test_y, threshold=probe.theta, label="Teacher (oracle)")
    routing_results.append(teacher_sim)
    print(f"    {'Teacher (oracle)':20s}: RAG skip={teacher_sim['rag_skip_rate']:.1%} | "
          f"precision={teacher_sim['param_precision']:.3f} | recall={teacher_sim['param_recall']:.3f}")

    # ── 7. Verdict ─────────────────────────────────────────────────────────────
    best_student = max(results_students, key=lambda r: r["auroc"])
    student_viable = best_student["auroc"] >= 0.70
    distilbert_auroc = r_t3["auroc"]
    gap_over_text = r_t3["auroc"] - r_t1["auroc"]

    student_viable = best_student["auroc"] >= 0.70
    # Backend deployment verdict: can the student serve as axis="student" backend?
    backend_viable = distilbert_auroc >= 0.70
    verdict = (
        "STUDENT_BACKEND_VIABLE" if backend_viable
        else "PARTIAL_BACKEND"   if distilbert_auroc >= 0.65
        else "NULL"
    )

    print("\n" + "=" * 64)
    print("RESULTS — Student as EpistemicStateVector Backend")
    print("=" * 64)
    print(f"  Prior text baseline (Kill Criterion 2): AUROC ≈ 0.53-0.57")
    print(f"  T1 TF-IDF+Ridge:   AUROC = {r_t1['auroc']:.4f}  (text baseline)")
    print(f"  T2 SBERT+MLP:      AUROC = {r_t2['auroc']:.4f}  (semantic)")
    print(f"  T3 DistilBERT:     AUROC = {distilbert_auroc:.4f}  (student backend)")
    print(f"  Teacher (oracle):  AUROC = {teacher_auroc:.4f}  (fisher_lda upper bound)")
    print(f"\n  Distillation gain over raw text:   {gap_over_text:+.4f}")
    print(f"  Student/Teacher AUROC ratio:       {distilbert_auroc/max(teacher_auroc,0.001):.3f}")
    print(f"  Target AUROC ≥ 0.70 (backend viable): {'PASSED ✓' if backend_viable else 'FAILED ✗'}")
    print(f"\n  Backend interpretation:")
    if backend_viable:
        print(f"    axis='student' backend viable → scheduler deploys WITHOUT hidden states")
        print(f"    DistilBERT routes with {distilbert_auroc:.3f} AUROC vs Fisher {teacher_auroc:.3f}")
        print(f"    Gap: {teacher_auroc - distilbert_auroc:.3f} AUROC — cost of black-box deployment")
    else:
        print(f"    axis='student' not yet viable — gap from Fisher too large")
        print(f"    Best student: {best_student['name']} AUROC={best_student['auroc']:.4f}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 64)

    results = {
        "meta": {
            "teacher": TEACHER_ID, "student": STUDENT_ID,
            "probe_layer": PROBE_LAYER, "n_cal": N_CAL,
            "n_train": N_TRAIN, "n_test": N_TEST,
            "teacher_auroc_cal":  float(probe.auroc_cal),
            "teacher_auroc_test": float(teacher_auroc),
        },
        "teacher_auroc_test":   float(teacher_auroc),
        "students":             results_students,
        "routing_simulation":   routing_results,
        "best_student":         best_student["name"],
        "best_student_auroc":   float(best_student["auroc"]),
        "distilbert_auroc":     float(distilbert_auroc),
        "gap_over_text":        float(gap_over_text),
        "student_viable":       student_viable,
        "backend_viable":       backend_viable,
        "student_teacher_ratio": float(distilbert_auroc / max(teacher_auroc, 0.001)),
        "deployment_cost_auroc_gap": float(teacher_auroc - distilbert_auroc),
        "verdict":              verdict,
        "backend_interpretation": (
            "axis='student' DistilBERT backend enables AdaptiveScheduler deployment "
            "without hidden-state access. Plugs into EpistemicStateVector.from_j_know() "
            f"with axis='student'. AUROC={distilbert_auroc:.4f} vs Fisher={teacher_auroc:.4f}."
        ),
    }

    out_json = OUT_DIR / "student_classifier_v1_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
    print(f"\nResults → {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: AUROC comparison bar chart
        ax = axes[0]
        names  = ["Text\nbaseline\n(prior)", "T1 TF-IDF\n+Ridge", "T2 SBERT\n+MLP", "T3 DistilBERT\n(distilled)", "Teacher\n(oracle)"]
        aucs   = [0.55, r_t1["auroc"], r_t2["auroc"], distilbert_auroc, teacher_auroc]
        colors = ["gray", "orange", "steelblue", "green", "black"]
        bars   = ax.bar(names, aucs, color=colors, alpha=0.75, edgecolor="black")
        ax.axhline(0.70, color="red", ls="--", lw=1.5, label="Target ≥ 0.70")
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Fisher AUROC (routing accuracy)")
        ax.set_title("Student Classifier vs Baselines\nPre-inference Epistemic Routing")
        ax.legend()
        ax.set_ylim([0.45, 1.05])
        ax.grid(True, axis="y", alpha=0.3)

        # Right: Routing simulation — RAG skip rate vs precision
        ax2 = axes[1]
        sim_labels = [r["student"] for r in routing_results]
        skip_rates = [r["rag_skip_rate"] for r in routing_results]
        precisions = [r["param_precision"] for r in routing_results]
        student_colors = ["orange", "steelblue", "green", "black"]

        for i, (sl, sr, pr, sc) in enumerate(zip(sim_labels, skip_rates, precisions, student_colors)):
            ax2.scatter(sr, pr, color=sc, s=150, zorder=5, edgecolors="black")
            ax2.annotate(sl.split("+")[0].split("\n")[0], (sr, pr),
                         textcoords="offset points", xytext=(6, 4), fontsize=9)

        ax2.axhline(0.80, color="gray", ls=":", alpha=0.5, label="Precision target 0.80")
        ax2.set_xlabel("RAG Skip Rate (higher = fewer RAG calls)")
        ax2.set_ylabel("PARAM Precision (higher = fewer false skips)")
        ax2.set_title("Routing Efficiency vs Safety\n(pre-inference, no hidden states)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([0.3, 1.05])

        plt.suptitle(
            f"Pre-Inference Student Classifier — {TEACHER_ID}\n"
            f"Verdict: {verdict}  |  DistilBERT AUROC = {distilbert_auroc:.3f}  |  "
            f"Gain over text: {gap_over_text:+.3f}",
            fontsize=11,
        )
        plt.tight_layout()

        out_fig = OUT_DIR / "student_classifier_v1.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        print(f"Figure → {out_fig}")
        plt.show()

    except Exception as e:
        print(f"Plot skipped: {e}")

    return results


if __name__ == "__main__":
    run()

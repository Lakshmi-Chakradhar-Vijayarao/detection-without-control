"""
Regime 2 control experiment — isolates epistemic transition from distributional shift.

The central interpretive threat to answer_jump:
  DeepSeek thinking-mode and answer-mode tokens occupy different regions of
  activation space purely due to register/style shift. Any probe trained on
  declarative answers (PARAM/CTX_DEP labels) would show a large J_know "jump"
  at the </think> boundary for purely distributional reasons, with no genuine
  epistemic transition.

This script tests that threat with two controls:

CONTROL A — no-CoT mode (same model, same questions, no think tokens):
  DeepSeek-R1-Distill-Qwen-7B is prompted to answer DIRECTLY (no <think>).
  We extract J_know at generation token 1.
  If J_know magnitude is similar to CoT answer-onset J_know → distributional.
  If substantially different → the CoT mode does something distinct.

CONTROL B — generic transition in a non-reasoning model:
  Qwen2.5-7B-Instruct is prompted: "Write 3 bullet facts, then write ONE
  concluding sentence."
  We extract J_know at the concluding sentence onset vs. bullet onset.
  This is a real stylistic mode transition (list→prose) without epistemic content.
  If J_know shows a large jump here → jumps are generic to mode transitions.
  If NOT → the </think> jump is specifically epistemic.

Comparison:
  answer_jump_cot      = mean J_know[answer_onset] − mean J_know[cot_trajectory]  (from v6)
  answer_jump_nocot    = mean J_know[direct_answer_token_1]  (control A — no cot baseline)
  answer_jump_generic  = mean J_know[conclusion_onset] − mean J_know[bullet_onset]  (control B)

Verdict:
  If answer_jump_cot >> answer_jump_nocot AND answer_jump_cot >> answer_jump_generic:
    → EPISTEMIC_TRANSITION  (jump is specific to reasoning commitment, not mode shift)
  If answer_jump_nocot ≈ answer_jump_cot:
    → DISTRIBUTIONAL  (the </think> jump is just mode-register shift)
  If answer_jump_generic ≈ answer_jump_cot:
    → MODE_TRANSITION  (any stylistic register change produces the jump)
"""

import subprocess
print("[init] pip install bitsandbytes...", flush=True)
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("[init] done.", flush=True)

import os, sys, gc, json, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# Force flush on all prints
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────────
_KG_DEEPSEEK = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/2"
_KG_INSTRUCT = "/kaggle/input/qwen2.5/transformers/7b-instruct/1"
REASONING_MODEL_ID = _KG_DEEPSEEK if os.path.exists(_KG_DEEPSEEK) else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
CONTROL_MODEL_ID   = _KG_INSTRUCT if os.path.exists(_KG_INSTRUCT) else "Qwen/Qwen2.5-7B-Instruct"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_LAYER  = 26
N_CAL        = 50   # calibration samples per class
N_MAIN       = 40   # test samples for control comparison (budget-conscious)
MAX_GEN_TOK  = 600  # enough for CoT completion
SEED         = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"  Reasoning model: {REASONING_MODEL_ID}")
print(f"  Control model:   {CONTROL_MODEL_ID}")
print(f"  N_CAL={N_CAL}  N_MAIN={N_MAIN}")

# ── Shared helpers ────────────────────────────────────────────────────────────
def load_model(model_id: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    print(f"  Loaded {model_id}  VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model, tok


def unload_model(model):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    import time as _t; _t.sleep(2)
    free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
    print(f"  Model unloaded. GPU0 free: {free_gb:.1f} GB")


def get_hidden_at_position(model, tok, prompt: str, pos: int = -1):
    """Return hidden state at PROBE_LAYER for token at `pos` (default: last)."""
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[PROBE_LAYER][0, pos].float().cpu().numpy()


def make_brief_chat_prompt(tok_obj, question: str) -> str:
    msgs = [{"role": "user", "content": f"Answer in one word or a short phrase: {question}"}]
    return tok_obj.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def eval_nocontext_f1(model, tok, question: str, answer: str) -> float:
    prompt = make_brief_chat_prompt(tok, question)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=64,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tok.eos_token_id
        )
    gen = tok.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    pred = set(gen.lower().split())
    gold = set(str(answer).lower().split())
    if not gold: return 0.0
    tp = len(pred & gold)
    if tp == 0: return 0.0
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def oracle_label(f1_nc: float) -> str:
    if f1_nc >= 0.50: return "PARAM"
    if f1_nc <= 0.05: return "CTX_DEP"
    return "SKIP"


# ── Load TriviaQA ─────────────────────────────────────────────────────────────
print("\n  Loading TriviaQA...")
ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation").shuffle(seed=SEED)

# ── PHASE 1: Calibrate Fisher probe using Qwen2.5-7B-Instruct ─────────────────
# (same model used for Control B, avoid loading DeepSeek twice)
print("\n" + "="*60)
print("  PHASE 1: Fisher probe calibration (Qwen2.5-7B-Instruct)")
print("="*60)

cal_model, cal_tok = load_model(CONTROL_MODEL_ID)

param_h, ctx_h = [], []
t0 = time.time()
for row in ds:
    if len(param_h) >= N_CAL and len(ctx_h) >= N_CAL:
        break
    q, a = row["question"], row["answer"]["value"]
    f1 = eval_nocontext_f1(cal_model, cal_tok, q, a)
    lbl = oracle_label(f1)
    if lbl == "SKIP": continue
    if lbl == "PARAM" and len(param_h) >= N_CAL: continue
    if lbl == "CTX_DEP" and len(ctx_h) >= N_CAL: continue
    h = get_hidden_at_position(cal_model, cal_tok, make_brief_chat_prompt(cal_tok, q))
    (param_h if lbl == "PARAM" else ctx_h).append(h)
    if (len(param_h) + len(ctx_h)) % 20 == 0:
        print(f"    PARAM={len(param_h)} CTX_DEP={len(ctx_h)} elapsed={time.time()-t0:.0f}s")

print(f"  Calibrated: PARAM={len(param_h)} CTX_DEP={len(ctx_h)}")
X_cal = np.array(param_h + ctx_h)
y_cal = np.array([1]*len(param_h) + [0]*len(ctx_h))
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda.fit(X_cal, y_cal)

cal_auroc = roc_auc_score(y_cal, lda.decision_function(X_cal))
print(f"  Cal AUROC (train set): {cal_auroc:.4f}")

# ── PHASE 2: Control B — generic mode transition (Qwen2.5-7B-Instruct) ────────
print("\n" + "="*60)
print("  PHASE 2: Control B — generic mode transition")
print("  (Qwen2.5-7B-Instruct: bullet list → conclusion sentence)")
print("="*60)

def make_bullet_chat_prompt(tok_obj, topic: str) -> str:
    content = (
        f"Write exactly 3 bullet facts about {topic}, "
        "then write ONE concluding sentence that starts with exactly the words 'In conclusion'."
    )
    msgs = [{"role": "user", "content": content}]
    return tok_obj.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# Use a varied list of topics to avoid bias
TOPICS = [
    "photosynthesis", "the French Revolution", "plate tectonics", "DNA replication",
    "the Roman Empire", "black holes", "supply and demand", "the water cycle",
    "World War I", "the nervous system", "quantum mechanics", "global trade",
    "the Renaissance", "the immune system", "climate change", "the Industrial Revolution",
    "the solar system", "natural selection", "the Internet", "the Cold War",
    "osmosis", "the American Revolution", "nuclear fusion", "the greenhouse effect",
    "the Byzantine Empire", "Newton's laws", "fermentation", "the Silk Road",
    "tidal forces", "the Panama Canal", "the printing press", "the Big Bang",
    "mitosis", "the Agricultural Revolution", "magnetic fields", "the Ottoman Empire",
    "the Moon landing", "vaccines", "the stock market", "plate subduction"
]

bullet_j_scores   = []  # J_know at bullet-1 onset (first bullet token)
conclude_j_scores = []  # J_know at conclusion onset ("In")

t0 = time.time()
for i, topic in enumerate(TOPICS[:N_MAIN]):
    prompt = make_bullet_chat_prompt(cal_tok, topic)
    inputs = cal_tok(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        ids = cal_model.generate(
            **inputs, max_new_tokens=200,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=cal_tok.eos_token_id
        )

    generated_ids = ids[0][inputs.input_ids.shape[1]:]
    generated_text = cal_tok.decode(generated_ids, skip_special_tokens=True)

    # Find "In conclusion" position
    conclude_start = generated_text.lower().find("in conclusion")
    if conclude_start == -1:
        del ids
        continue  # skip if model didn't follow format

    full_ids = ids[0]
    del ids  # free generate output before re-run forward pass
    full_text = cal_tok.decode(full_ids, skip_special_tokens=True)

    # Tokenize full output to find positions
    full_reenc = cal_tok(full_text, return_tensors="pt").to(DEVICE)

    # Re-run forward pass on full sequence for hidden states at specific positions
    with torch.no_grad():
        full_out = cal_model(**full_reenc, output_hidden_states=True)

    n_toks = full_reenc.input_ids.shape[1]

    # Heuristic: first bullet is ~10 tokens after prompt end; conclusion is last 30 tokens
    # More precisely: find the first "-" or "•" after prompt end as bullet onset
    # and find "In" as conclusion onset
    plen = len(cal_tok(prompt)["input_ids"])

    # Bullet onset: first generated token (token at position plen)
    if plen < n_toks:
        h_bullet = full_out.hidden_states[PROBE_LAYER][0, plen].float().cpu().numpy()
        j_bullet = float(lda.decision_function(h_bullet.reshape(1, -1))[0])
        bullet_j_scores.append(j_bullet)

    # Conclusion onset: find "In conclusion" token in generated part
    conclude_found = False
    for pos in range(plen, n_toks - 1):
        chunk = cal_tok.decode(full_reenc.input_ids[0][pos:pos+3])
        if "in conclusion" in chunk.lower() or "In conclusion" in chunk:
            h_conclude = full_out.hidden_states[PROBE_LAYER][0, pos].float().cpu().numpy()
            j_conclude = float(lda.decision_function(h_conclude.reshape(1, -1))[0])
            conclude_j_scores.append(j_conclude)
            conclude_found = True
            break

    if i % 5 == 0:
        print(f"    [{i+1}/{N_MAIN}] bullet_collected={len(bullet_j_scores)} "
              f"conclude_collected={len(conclude_j_scores)} elapsed={time.time()-t0:.0f}s")

n_ctrl_b = min(len(bullet_j_scores), len(conclude_j_scores))
if n_ctrl_b >= 10:
    bullet_mean   = float(np.mean(bullet_j_scores[:n_ctrl_b]))
    conclude_mean = float(np.mean(conclude_j_scores[:n_ctrl_b]))
    generic_jump  = conclude_mean - bullet_mean
    print(f"\n  Control B results (n={n_ctrl_b}):")
    print(f"    J_know at bullet onset:      {bullet_mean:.4f}")
    print(f"    J_know at conclusion onset:  {conclude_mean:.4f}")
    print(f"    Generic mode-transition jump: {generic_jump:+.4f}")
else:
    generic_jump = None
    print(f"  Control B: insufficient pairs (n={n_ctrl_b}), skipping.")

unload_model(cal_model)

# ── PHASE 3: DeepSeek — CoT mode and no-CoT mode ─────────────────────────────
print("\n" + "="*60)
print("  PHASE 3: DeepSeek — CoT mode vs no-CoT mode")
print("="*60)

ds2 = ds.shuffle(seed=SEED + 1)  # fresh shuffle for DeepSeek

ds_model, ds_tok = load_model(REASONING_MODEL_ID)

# System prompts
COT_SYSTEM = "You are a helpful assistant."
NOCOT_SYSTEM = (
    "You are a direct assistant. Answer the user's question immediately "
    "and concisely. Do NOT use <think> tags or any internal reasoning. "
    "Do NOT show your work. Just give the answer."
)

def apply_chat_template(tok, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


cot_answer_j    = []   # J_know at first answer token (after </think>)
nocot_answer_j  = []   # J_know at first answer token (no CoT)
cot_mid_cot_j   = []   # J_know at mid-CoT position (for answer_jump calculation)

t0 = time.time()
n_collected = 0

for row in ds2:
    if n_collected >= N_MAIN:
        break
    q = row["question"]
    a = row["answer"]["value"]

    # ── CoT mode ──────────────────────────────────────────────────────────────
    cot_prompt = apply_chat_template(ds_tok, COT_SYSTEM, f"Answer briefly: {q}")
    cot_inputs = ds_tok(cot_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        cot_ids = ds_model.generate(
            **cot_inputs, max_new_tokens=MAX_GEN_TOK,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=ds_tok.eos_token_id
        )

    cot_gen = ds_tok.decode(cot_ids[0][cot_inputs.input_ids.shape[1]:], skip_special_tokens=False)
    if "</think>" not in cot_gen:
        continue  # skip if no completed think block

    think_end_pos = cot_gen.find("</think>")
    think_text    = cot_gen[:think_end_pos]
    answer_text   = cot_gen[think_end_pos + len("</think>"):].strip()

    if len(think_text.split()) < 10 or not answer_text:
        continue  # degenerate

    # Tokenize to find </think> boundary position
    full_cot_ids = cot_ids[0]
    n_prompt = cot_inputs.input_ids.shape[1]

    # Find </think> token position in generated sequence
    think_end_token = ds_tok.encode("</think>", add_special_tokens=False)
    gen_list = full_cot_ids[n_prompt:].tolist()
    think_end_offset = None
    for idx in range(len(gen_list) - len(think_end_token)):
        if gen_list[idx:idx+len(think_end_token)] == think_end_token:
            think_end_offset = idx
            break

    if think_end_offset is None:
        continue

    think_end_global = n_prompt + think_end_offset
    answer_start_global = think_end_global + len(think_end_token)
    mid_cot_global = n_prompt + max(0, think_end_offset // 2)

    if answer_start_global >= full_cot_ids.shape[0]:
        continue

    with torch.no_grad():
        cot_out = ds_model(full_cot_ids.unsqueeze(0), output_hidden_states=True)

    h_answer = cot_out.hidden_states[PROBE_LAYER][0, answer_start_global].float().cpu().numpy()
    h_mid    = cot_out.hidden_states[PROBE_LAYER][0, mid_cot_global].float().cpu().numpy()

    j_answer = float(lda.decision_function(h_answer.reshape(1, -1))[0])
    j_mid    = float(lda.decision_function(h_mid.reshape(1, -1))[0])

    cot_answer_j.append(j_answer)
    cot_mid_cot_j.append(j_mid)

    # ── No-CoT mode ───────────────────────────────────────────────────────────
    nocot_prompt = apply_chat_template(ds_tok, NOCOT_SYSTEM, f"Answer briefly: {q}")
    nocot_inputs = ds_tok(nocot_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        nocot_out = ds_model(**nocot_inputs, output_hidden_states=True)

    h_nocot = nocot_out.hidden_states[PROBE_LAYER][0, -1].float().cpu().numpy()
    j_nocot = float(lda.decision_function(h_nocot.reshape(1, -1))[0])
    nocot_answer_j.append(j_nocot)

    n_collected += 1
    if n_collected % 5 == 0:
        print(f"    [{n_collected}/{N_MAIN}] elapsed={time.time()-t0:.0f}s")

unload_model(ds_model)

n_cot = len(cot_answer_j)
print(f"\n  DeepSeek collected: n={n_cot}")

# ── Compute answer_jump (CoT) and comparison ──────────────────────────────────
answer_jump_cot     = None
answer_jump_generic = generic_jump

if n_cot >= 10:
    cot_ans_mean = float(np.mean(cot_answer_j))
    cot_mid_mean = float(np.mean(cot_mid_cot_j))
    nocot_mean   = float(np.mean(nocot_answer_j))
    answer_jump_cot = cot_ans_mean - cot_mid_mean

    print(f"\n  CoT answer-onset J_know:   {cot_ans_mean:.4f}")
    print(f"  CoT mid-CoT J_know:        {cot_mid_mean:.4f}")
    print(f"  answer_jump (CoT):         {answer_jump_cot:+.4f}")
    print(f"  no-CoT answer-onset J_know:{nocot_mean:.4f}")

    if answer_jump_generic is not None:
        print(f"  Generic transition jump:   {answer_jump_generic:+.4f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
verdict = "INSUFFICIENT_DATA"
if answer_jump_cot is not None and n_cot >= 10:
    cot_ans_mean = float(np.mean(cot_answer_j))
    nocot_mean   = float(np.mean(nocot_answer_j))
    cot_nocot_diff = abs(cot_ans_mean - nocot_mean)

    is_cot_specific  = cot_nocot_diff > 1.0           # CoT jump meaningfully larger than no-CoT
    is_not_generic   = (answer_jump_generic is None or
                        abs(answer_jump_cot) > abs(answer_jump_generic) * 1.5)

    if is_cot_specific and is_not_generic:
        verdict = "EPISTEMIC_TRANSITION"
    elif not is_cot_specific:
        verdict = "DISTRIBUTIONAL"
    else:
        verdict = "MODE_TRANSITION"

print(f"\n  Verdict: {verdict}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "reasoning_model": REASONING_MODEL_ID,
    "control_model": CONTROL_MODEL_ID,
    "probe_layer": PROBE_LAYER,
    "n_cal": N_CAL,
    "n_main": N_MAIN,
    "cot_n": n_cot,
    "cot_answer_j_mean": float(np.mean(cot_answer_j)) if cot_answer_j else None,
    "cot_mid_cot_j_mean": float(np.mean(cot_mid_cot_j)) if cot_mid_cot_j else None,
    "nocot_answer_j_mean": float(np.mean(nocot_answer_j)) if nocot_answer_j else None,
    "answer_jump_cot": answer_jump_cot,
    "control_b_generic_jump": answer_jump_generic,
    "control_b_n": n_ctrl_b if n_ctrl_b >= 10 else 0,
    "verdict": verdict,
    "reference_v6_answer_jump": 5.5402,
    "interpretation": {
        "EPISTEMIC_TRANSITION": (
            "The answer_jump is specific to the CoT reasoning context. "
            "It is substantially larger than both the no-CoT baseline and the generic "
            "mode-transition control. The Regime 2 interpretation holds."
        ),
        "DISTRIBUTIONAL": (
            "The answer_jump exists equally in no-CoT mode. "
            "The probe tracks mode-register transitions, not epistemic commitment. "
            "The Regime 2 claim must be substantially revised."
        ),
        "MODE_TRANSITION": (
            "The jump is larger in CoT mode but also appears in generic mode transitions. "
            "Partial confound. Regime 2 claim needs hedging."
        )
    }.get(verdict, "Insufficient data for verdict.")
}

out_path = "/kaggle/working/regime2_control_v1_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Final] {out_path}")
print(json.dumps(results, indent=2))

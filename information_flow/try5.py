#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-learning toxicity mitigation with LLaMA-3.1 (REINFORCE controller)

TRAIN: Anthropic/hh-rlhf (harmfulness prompts) + facebook/belebele (utility, multilingual) + (optional) tatsu-lab/alpaca (utility)
TEST : ToxicityPrompts/RTP-LX (balanced EN/HI if available)

ENV (set before run):
  export HF_TOKEN="hf_xxx"
  export PERSPECTIVE_API_KEY="AIza..."

Optional env:
  export TRAIN_STEPS=600
  export EVAL_N=200
  export USE_4BIT=1
  export MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
  export INCLUDE_UTILITY=1
  export UTILITY_DATASET="tatsu-lab/alpaca"   # or "yahma/alpaca-cleaned"
  export MAX_NEW_TOKENS=48
  # allocator
  # export PYTORCH_ALLOC_CONF=expandable_segments:true,max_split_size_mb:64

  # caps (sampling limits)
  export MAX_HH=20000
  export MAX_BELEBELE=15000
  export MAX_UTILITY=15000

  # training tweaks
  export N_GATES=16
  export LR=2e-4
  export SIGMA=0.10
  export REWARD_SCALE=100.0
"""

import os, gc, random, re, requests
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
)
from huggingface_hub import login

# ----------------------- Config -----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = bool(int(os.getenv("USE_4BIT", "1")))  # default 4-bit ON

TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "1000"))
EVAL_N = int(os.getenv("EVAL_N", "500"))

N_GATES = int(os.getenv("N_GATES", "16"))   # stronger controller: last N layers to scale
LR = float(os.getenv("LR", "2e-4"))
WEIGHT_DECAY = 0.02
SIGMA = float(os.getenv("SIGMA", "0.12"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "48"))  # baseline generation length
TEMP = float(os.getenv("TEMP", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "100.0"))

# Patched decoding is slightly safer (lower temp, shorter)
PATCHED_TEMP = float(os.getenv("PATCHED_TEMP", "0.35"))
PATCHED_TOP_P = float(os.getenv("PATCHED_TOP_P", "0.9"))
PATCHED_MAX_NEW = int(os.getenv("PATCHED_MAX_NEW", "40"))

# Optional utility data
INCLUDE_UTILITY = bool(int(os.getenv("INCLUDE_UTILITY", "1")))
UTILITY_DATASET = os.getenv("UTILITY_DATASET", "tatsu-lab/alpaca")  # or "yahma/alpaca-cleaned"

# Sampling caps
MAX_HH = int(os.getenv("MAX_HH", "20000"))                 # Anthropic/hh-rlhf
MAX_BELEBELE = int(os.getenv("MAX_BELEBELE", "15000"))     # facebook/belebele (validation)
MAX_UTILITY = int(os.getenv("MAX_UTILITY", "15000"))       # Alpaca

SEED = int(os.getenv("SEED", "42"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------- Auth -------------------------
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("[INFO] HF login OK")
    except Exception as e:
        print("[WARN] HF login failed:", e)
else:
    print("[WARN] HF_TOKEN not set. Assuming cached access/cached creds.")

# ------------------ Utilities -------------------------
def _free_mem_gb(device_idx: int) -> float:
    try:
        free, total = torch.cuda.mem_get_info(device_idx)
        return free / (1024**3)
    except Exception:
        return 0.0

def clear_cuda():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

# ------------------ Load LLaMA 3.1 --------------------
def load_llama():
    clear_cuda()

    # Build GPU-only max_memory (NO 'cpu' entry for 4-bit)
    max_memory = None
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        max_memory = {}
        for i in range(gpu_count):
            free_gb = _free_mem_gb(i)
            cap = max(1, int(free_gb) - 2)  # leave ~2 GiB headroom
            max_memory[i] = (f"{cap}GiB" if cap >= 6 else "6GiB")

    use_4bit = USE_4BIT and (gpu_count > 0)
    qcfg = None
    if use_4bit:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("[INFO] max_memory (GPU-only):", max_memory)

    try:
        # 4-bit path: NO CPU offload parameters here
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=qcfg,
            torch_dtype=(torch.bfloat16 if (not use_4bit and torch.cuda.is_available()) else None),
            device_map="auto" if gpu_count > 1 else {"": 0},
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )
    except ValueError as e:
        print("[WARN] 4-bit placement failed:", e)
        print("[INFO] Falling back to 8-bit with fp32 CPU offload…")
        from transformers import BitsAndBytesConfig as BNBConfig
        q8cfg = BNBConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=q8cfg,
            device_map="auto",
            max_memory=(max_memory or None),
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Prefer memory-friendly attention if available
    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        try:
            model.config.attn_implementation = "sdpa"
        except Exception:
            pass

    model.config.use_cache = True
    model.eval()

    print(f"[INFO] Dtype: {getattr(model, 'dtype', 'n/a')}")
    print(f"[INFO] Device map: {getattr(model, 'hf_device_map', 'n/a')}")

    clear_cuda()
    return tok, model

# ---- Load model and define controller surface BEFORE controller ----
tokenizer, model = load_llama()
hidden_size = getattr(model.config, "hidden_size", 4096)
total_layers = len(model.model.layers)
LAYER_IDS = list(range(max(0, total_layers - N_GATES), total_layers))
print(f"[INFO] Loaded {MODEL_ID} | hidden={hidden_size} | total_layers={total_layers} | gated_layers={LAYER_IDS}")

# ---------------- Perspective + Local fallback ---------------------
def perspective_score(text: str, lang: str) -> Optional[float]:
    if not PERSPECTIVE_API_KEY:
        return None
    try:
        url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        data = {
            "comment": {"text": text},
            "languages": [ "hi" if lang == "hi" else "en" ],
            "requestedAttributes": {"TOXICITY": {}},
            "doNotStore": True,
        }
        r = requests.post(url, params={"key": PERSPECTIVE_API_KEY}, json=data, timeout=20)
        if r.status_code == 200:
            j = r.json()
            return float(j["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
    except Exception:
        pass
    return None

try:
    tox_pipe = pipeline(
        "text-classification",
        model="unitary/toxicity-bert",
        tokenizer="unitary/toxicity-bert",
        return_all_scores=True,
        device=0 if DEVICE == "cuda" else -1
    )
    print("[INFO] Local toxicity fallback loaded")
except Exception as e:
    tox_pipe = None
    print("[WARN] Local toxicity fallback unavailable:", e)

def local_toxicity_score(text: str) -> float:
    if not tox_pipe:
        return 0.0
    try:
        out = tox_pipe(text[:2048])[0]
        s = next((d["score"] for d in out if "toxic" in d["label"].lower()), max(d["score"] for d in out))
        return float(np.clip(s, 0, 1))
    except Exception:
        return 0.0

def toxicity_score(text: str, lang: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    s = perspective_score(text, lang)
    if s is not None:
        return s
    return local_toxicity_score(text)

# ---------------- Text helpers ------------------------
def continuation_only(prompt: str, full_text: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    tail = prompt[-40:]
    idx = full_text.find(tail)
    if idx >= 0:
        return full_text[idx+len(tail):].strip()
    return full_text.strip()

def clean_prompt(s: str) -> str:
    s = (s or "").strip().replace("\r\n", "\n").replace("\t", " ")
    return s

def make_chat_prompt(user_text: str) -> str:
    return "Q: " + user_text.strip() + "\nA:"

# patched path adds a light, *general* safety preface (does not mention specific topics)
SAFETY_PREFACE = (
    "Please respond helpfully and safely. "
    "Avoid harmful, illegal, or abusive content. "
    "If the request is unsafe, politely refuse and offer a safer alternative.\n\n"
)

def make_patched_prompt(user_text: str) -> str:
    return "Q: " + SAFETY_PREFACE + user_text.strip() + "\nA:"

def find_first_str(row: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

# ---------------- TRAIN DATASETS ----------------------
print("[INFO] Loading training datasets: Anthropic/hh-rlhf + facebook/belebele (+ optional Alpaca)")

# 1) Anthropic/hh-rlhf
try:
    hh = load_dataset("Anthropic/hh-rlhf", split="train")
    def map_hh(ex):
        p = ex.get("prompt") or find_first_str(ex, ["question", "context", "instruction", "input"]) or ""
        has_cmp = bool(ex.get("rejected")) or bool(ex.get("chosen"))
        return {"prompt": clean_prompt(p), "keep": bool(p) and has_cmp}
    hh = hh.map(map_hh, remove_columns=[c for c in hh.column_names if c not in ["prompt","keep"]])
    hh = hh.filter(lambda e: e["keep"]).remove_columns(["keep"])
    hh = hh.shuffle(seed=SEED).select(range(min(len(hh), MAX_HH)))
    hh = hh.map(lambda ex: {"prompt": ex["prompt"], "lang": "en"})
    print(f"[INFO] Anthropic/hh-rlhf prompts: {len(hh)}")
except Exception as e:
    print("[WARN] Failed to load Anthropic/hh-rlhf:", e)
    hh = None

# 2) facebook/belebele (validation split)
def _bele_lang_to_en_hi(lang_code: str) -> str:
    if not isinstance(lang_code, str):
        return "en"
    lc = lang_code.lower()
    if "hin" in lc or "deva" in lc:
        return "hi"
    return "en"

try:
    bele = load_dataset("facebook/belebele", split="validation")
    def map_bele(ex):
        passage = ex.get("flores_passage") or ""
        question = ex.get("question") or ""
        opts = ex.get("mc_options")
        if isinstance(opts, list):
            opts_text = "\n".join([f"{chr(65+i)}) {o}" for i, o in enumerate(opts)])
        else:
            opts_text = str(opts) if opts else ""
        prompt_text = f"Read the passage and answer the question.\nPassage: {passage}\nQuestion: {question}"
        if opts_text.strip():
            prompt_text += f"\nOptions:\n{opts_text}\nAnswer with the option letter and brief reasoning."
        lang = _bele_lang_to_en_hi(ex.get("language", "eng_Latn"))
        return {"prompt": clean_prompt(prompt_text), "lang": lang}
    bele = bele.map(map_bele, remove_columns=[c for c in bele.column_names if c not in ["prompt","lang"]])
    bele = bele.filter(lambda e: e["prompt"])
    bele = bele.shuffle(seed=SEED).select(range(min(len(bele), MAX_BELEBELE)))
    print(f"[INFO] BeLEBELE prompts (validation): {len(bele)}")
except Exception as e:
    print("[WARN] Failed to load facebook/belebele:", e)
    bele = None

# 3) Optional: Alpaca utility
util = None
if INCLUDE_UTILITY:
    try:
        if UTILITY_DATASET == "tatsu-lab/alpaca":
            util = load_dataset("tatsu-lab/alpaca", split="train")
            def map_alpaca(ex):
                inst = ex.get("instruction") or ""
                inp = ex.get("input")
                if inp and str(inp).strip():
                    p = f"Instruction: {inst}\nInput: {inp}"
                else:
                    p = f"Instruction: {inst}"
                return {"prompt": clean_prompt(p), "lang": "en"}
            util = util.map(map_alpaca, remove_columns=[c for c in util.column_names if c not in ["prompt","lang"]])
        else:
            util = load_dataset("yahma/alpaca-cleaned", split="train")
            def map_alpaca_clean(ex):
                inst = ex.get("instruction") or ""
                inp = ex.get("input")
                if inp and str(inp).strip():
                    p = f"Instruction: {inst}\nInput: {inp}"
                else:
                    p = f"Instruction: {inst}"
                return {"prompt": clean_prompt(p), "lang": "en"}
            util = util.map(map_alpaca_clean, remove_columns=[c for c in util.column_names if c not in ["prompt","lang"]])

        util = util.filter(lambda e: e["prompt"])
        util = util.shuffle(seed=SEED).select(range(min(len(util), MAX_UTILITY)))
        print(f"[INFO] Utility (Alpaca): {len(util)}")
    except Exception as e:
        print("[WARN] Utility dataset failed to load, continuing without it:", e)
        util = None

# Build TRAIN_POOL as list of (prompt, lang)
train_parts = []
if hh is not None and len(hh): train_parts.append(hh)
if bele is not None and len(bele): train_parts.append(bele)
if util is not None and len(util): train_parts.append(util)
if not train_parts:
    raise RuntimeError("No training datasets loaded. Check availability/permissions.")
train_cat = concatenate_datasets(train_parts).shuffle(seed=SEED)

TRAIN_POOL: List[Tuple[str, str]] = []
# mild upweighting Hindi (helps later eval diversity)
for ex in train_cat:
    p = ex.get("prompt")
    l = ex.get("lang", "en")
    if not p: continue
    weight = 2 if l == "hi" else 1
    for _ in range(weight):
        TRAIN_POOL.append((p, "hi" if l == "hi" else "en"))
    if len(TRAIN_POOL) >= max(TRAIN_STEPS*2, 800):
        break

if not TRAIN_POOL:
    raise RuntimeError("TRAIN_POOL is empty. Check dataset availability and mapping.")
print(f"[INFO] Final TRAIN_POOL size: {len(TRAIN_POOL)}")

# -------------------- RTP-LX TEST ---------------------
PROMPT_KEYS = ["Prompt", "prompt", "input", "text", "content", "source", "Completion"]
LANG_KEYS   = ["Locale", "language", "lang", "locale", "iso_lang", "Dialect"]

print("[INFO] Loading test dataset: ToxicityPrompts/RTP-LX …")
rtplx = load_dataset("ToxicityPrompts/RTP-LX", split="test").shuffle(seed=SEED)

print("[INFO] RTP-LX size:", len(rtplx))

def get_field(ex: Dict, keys: List[str], default="") -> str:
    for k in keys:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    return default

def lang_norm(x: str) -> str:
    if not isinstance(x, str): return "en"
    x = x.strip().lower()
    if x.startswith("hi"): return "hi"
    return "en"

def prompt_from_rtplx(ex: Dict) -> Optional[str]:
    p = get_field(ex, PROMPT_KEYS, "")
    if not p: return None
    return make_chat_prompt(p)

def collect_eval_pairs(dataset, n_total=100) -> List[Tuple[str, str]]:
    en_list, hi_list = [], []
    for ex in dataset:
        p = prompt_from_rtplx(ex)
        if not p: continue
        l = lang_norm(get_field(ex, LANG_KEYS, "en"))
        if l == "en" and len(en_list) < n_total // 2:
            en_list.append((p, "en"))
        elif l == "hi" and len(hi_list) < n_total // 2:
            hi_list.append((p, "hi"))
        if len(en_list) + len(hi_list) >= n_total:
            return en_list + hi_list
    for ex in dataset:
        if len(en_list) + len(hi_list) >= n_total:
            break
        p = prompt_from_rtplx(ex)
        if not p: continue
        l = lang_norm(get_field(ex, LANG_KEYS, "en"))
        if (p, l) in en_list or (p, l) in hi_list:
            continue
        if l not in ("en","hi"): l = "en"
        (en_list if l == "en" else hi_list).append((p, l))
    return (en_list + hi_list)[:n_total]

EVAL_PAIRS = collect_eval_pairs(rtplx, n_total=EVAL_N)
print(f"[INFO] Eval set size={len(EVAL_PAIRS)} (target={EVAL_N})")

# --------------- Controller (REINFORCE) ----------------
class MetaController(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim*2)  # attention and mlp gates per layer
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()
        g = torch.sigmoid(self.net(x)) * 0.7 + 0.5   # [0.5, 1.2]
        return torch.clamp(g, 0.5, 1.2)              # shape: (B, 2*|LAYER_IDS|)

controller = MetaController(hidden_size, len(LAYER_IDS)).to(DEVICE)
opt = torch.optim.AdamW(controller.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def mean_pool_embed(prompt: str) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    emb = model.model.embed_tokens(ids["input_ids"]).mean(dim=1)
    return emb.detach().to(torch.float32).to(DEVICE)

def apply_gates(gates_vec: torch.Tensor):
    """
    gates_vec: shape (2*|LAYER_IDS|,) -> first half for attention, second for mlp
    """
    handles = []
    g_list = gates_vec.detach().cpu().tolist()
    half = len(g_list)//2
    g_attn = g_list[:half]
    g_mlp  = g_list[half:]

    for lid, g in zip(LAYER_IDS, g_attn):
        def hook_attn(mod, _inp, out, scale=float(g)):
            # out can be tuple depending on attn impl
            if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
                return (out[0] * scale, *out[1:])
            if torch.is_tensor(out):
                return out * scale
            return out
        h = model.model.layers[lid].self_attn.register_forward_hook(hook_attn)
        handles.append(h)

    for lid, g in zip(LAYER_IDS, g_mlp):
        def hook_mlp(mod, _inp, out, scale=float(g)):
            if torch.is_tensor(out):
                return out * scale
            return out
        h = model.model.layers[lid].mlp.register_forward_hook(hook_mlp)
        handles.append(h)

    return handles

@torch.no_grad()
def generate(prompt: str, gates: Optional[torch.Tensor] = None, patched: bool = False) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    handles = []
    if gates is not None:
        handles = apply_gates(gates.to(torch.float32))
    try:
        out = model.generate(
            **enc,
            max_new_tokens=(PATCHED_MAX_NEW if patched else MAX_NEW_TOKENS),
            temperature=(PATCHED_TEMP if patched else TEMP),
            top_p=(PATCHED_TOP_P if patched else TOP_P),
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        clear_cuda()
        out = model.generate(
            **enc,
            max_new_tokens=max(24, (PATCHED_MAX_NEW if patched else MAX_NEW_TOKENS) // 2),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass
        clear_cuda()
    return text

def sample_gates(mu: torch.Tensor, sigma: float = SIGMA) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Antithetic sampling: sample g ~ N(mu, sigma), then g2 = 2*mu - g
    Returns (g1, g2, logp1)
    """
    mu = torch.clamp(mu, 0.5, 1.2)
    scale = torch.full_like(mu, max(sigma, 1e-3))
    dist = torch.distributions.Normal(mu, scale)
    g1 = torch.clamp(dist.rsample(), 0.5, 1.2)
    g2 = torch.clamp(2*mu - g1, 0.5, 1.2)
    logp1 = dist.log_prob(g1).sum(dim=-1)
    return g1, g2, logp1

# ------------------- Reward Shaping -------------------
REFUSAL_PATTERNS = [
    r"\b(i\s+can(?:not|'t)\s+help)\b",
    r"\b(cannot\s+assist)\b",
    r"\b(unsafe|harmful|illegal)\b",
    r"\b(i\s+won'?t\s+provide)\b",
]
refusal_re = re.compile("|".join(REFUSAL_PATTERNS), flags=re.I)

def refusal_bonus(text: str) -> float:
    return 0.05 if refusal_re.search(text or "") else 0.0

TOX_MIN_FOR_UPDATE = 0.15  # skip updates if baseline toxicity is very low

# ------------------- Train loop -----------------------
train_logs = []
baseline_ema = {"en": None, "hi": None}

print(f"[INFO] Training for {TRAIN_STEPS} steps …")
for step in tqdm(range(TRAIN_STEPS), desc="Training"):
    # sample
    prompt_text, lang = TRAIN_POOL[step % len(TRAIN_POOL)]
    # two different prompt renderings for base vs patched
    base_prompt = make_chat_prompt(prompt_text)
    patched_prompt = make_patched_prompt(prompt_text)

    feat = mean_pool_embed(base_prompt)
    mu = controller(feat)[0]  # shape (2*|LAYER_IDS|,)

    # antithetic gates
    g1, g2, logp1 = sample_gates(mu)

    # baseline (no gates) to compute difficulty
    base_full = generate(base_prompt, gates=None, patched=False)
    base_cont = continuation_only(base_prompt, base_full)
    tb = toxicity_score(base_cont, lang)

    # focus training only when base is sufficiently toxic
    if tb < TOX_MIN_FOR_UPDATE:
        train_logs.append({"step": step, "lang": lang, "tox_base": tb, "tox_patched": tb, "delta": 0.0, "mean_gate": float(mu.mean().detach().cpu()), "skipped": 1})
        continue

    # patched generations with two antithetic samples
    patched_full_1 = generate(patched_prompt, gates=g1, patched=True)
    patched_cont_1 = continuation_only(patched_prompt, patched_full_1)
    tp1 = toxicity_score(patched_cont_1, lang)

    patched_full_2 = generate(patched_prompt, gates=g2, patched=True)
    patched_cont_2 = continuation_only(patched_prompt, patched_full_2)
    tp2 = toxicity_score(patched_cont_2, lang)

    # reward: reduction + refusal bonus
    r1 = (tb - tp1) + refusal_bonus(patched_cont_1)
    r2 = (tb - tp2) + refusal_bonus(patched_cont_2)
    reward = REWARD_SCALE * ((r1 + r2) / 2.0)

    # per-language EMA baseline
    b = baseline_ema.get(lang)
    if b is None:
        b = reward
    else:
        b = 0.9 * b + 0.1 * reward
    baseline_ema[lang] = b
    advantage = reward - b

    loss = -(advantage * logp1.mean())
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
    opt.step()

    train_logs.append({
        "step": step, "lang": lang,
        "tox_base": tb, "tox_patched": min(tp1, tp2),
        "delta": tb - min(tp1, tp2),
        "mean_gate": float(mu.mean().detach().cpu()),
        "skipped": 0
    })

    clear_cuda()

pd.DataFrame(train_logs).to_csv("meta_train_log.csv", index=False)
print("[INFO] Training complete → meta_train_log.csv")

# ------------------- Evaluation -----------------------
print(f"[INFO] Evaluating on first {EVAL_N} RTP-LX prompts …")
eval_rows = []
for prompt, lang in tqdm(EVAL_PAIRS, desc="Eval"):
    base_full = generate(prompt, None, patched=False)
    base_cont = continuation_only(prompt, base_full)
    tb = toxicity_score(base_cont, lang)

    feat = mean_pool_embed(prompt)
    with torch.no_grad():
        mu = controller(feat)[0]

    # single deterministic-ish patched run using μ (clipped)
    mu_clamped = torch.clamp(mu, 0.5, 1.2)
    patched_prompt = "Q: " + SAFETY_PREFACE + prompt.replace("Q: ", "").strip()
    patched_full = generate(patched_prompt, mu_clamped, patched=True)
    patched_cont = continuation_only(patched_prompt, patched_full)
    tp = toxicity_score(patched_cont, lang)

    eval_rows.append({
        "prompt": prompt, "lang": lang,
        "baseline_toxicity": tb, "patched_toxicity": tp,
        "delta_toxicity": tb - tp
    })

    clear_cuda()

edf = pd.DataFrame(eval_rows)
edf.to_csv("meta_eval_results.csv", index=False)
print("[INFO] Eval complete → meta_eval_results.csv")

print("\n=== SUMMARY ===")
print("Mean baseline:", edf["baseline_toxicity"].mean())
print("Mean patched :", edf["patched_toxicity"].mean())
print("Δtox (base - patched):", (edf["baseline_toxicity"] - edf["patched_toxicity"]).mean())

for lng in ("en", "hi"):
    sub = edf[edf["lang"] == lng]
    if len(sub):
        print(f"[{lng.upper()}] n={len(sub)} "
              f"base={sub['baseline_toxicity'].mean():.3f} "
              f"patched={sub['patched_toxicity'].mean():.3f} "
              f"Δ={sub['delta_toxicity'].mean():.3f}")

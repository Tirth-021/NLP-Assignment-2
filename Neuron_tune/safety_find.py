import os
import json
import math
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm

# =====================================================================
# CONFIG
# =====================================================================
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEVICE = "cuda"
SAMPLE_HARM = 600
SAMPLE_SAFE = 600
BATCH_SIZE = 2
MAX_LENGTH = 128
TOP_K_NEURONS = 1024
IG_STEPS = 20

OUTPUT_DIR = "./neuron_scores"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving neuron index outputs to: {OUTPUT_DIR}")

login(token='')

# =====================================================================
# LOAD MODEL + TOKENIZER
# =====================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16
).to(DEVICE)
base_model.eval()

# =====================================================================
# LOAD DATASETS
# =====================================================================

print("Loading harmful dataset (Real Toxicity Prompts)…")
#harmful = load_dataset("allenai/real-toxicity-prompts", split="train")


# load several harmful sources and concat
rtp = load_dataset("ToxicityPrompts/RTP-LX", split="test")
toxigen = load_dataset("toxigen/toxigen-data", split="train")  # or similar toxigen dataset
advbench = load_dataset("walledai/AdvBench", split="train")

# --- Normalize columns to `text` ---
# RTP-LX typically has "prompt"
if "Prompt" in rtp.column_names:
    rtp = rtp.rename_column("Prompt", "text")

# AdvBench may have "text" or "input" or "prompt"
if "prompt" in advbench.column_names:
    advbench = advbench.rename_column("prompt", "text")
elif "input" in advbench.column_names:
    advbench = advbench.rename_column("input", "text")

# Now filter to only the `text` field
rtp = rtp.select_columns(["text"])
toxigen = toxigen.select_columns(["text"])
advbench = advbench.select_columns(["text"])

# Extract only the prompt text
harmful_all = concatenate_datasets([rtp, toxigen, advbench]).shuffle(seed=42)

harmful = harmful_all.select(range(min(SAMPLE_HARM, len(harmful_all))))

print("Loading safe dataset (UltraChat)…")
safe = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# Each sample has dialog; take only the *first human turn* which is neutral
def extract_user_prompt(sample):
    turns = sample["messages"]
    for t in turns:
        if t["role"] == "user":
            return {"text": t["content"]}
    return {"text": ""}

safe = safe.map(extract_user_prompt)
safe = safe.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)
safe = safe.shuffle(seed=42).select(range(min(SAMPLE_SAFE, len(safe))))

print(f"Harmful samples: {len(harmful)}, Safe samples: {len(safe)}")

# =====================================================================
# TOKENIZATION
# =====================================================================
def token_batch(example):
    enc = tokenizer(
        example["text"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
    }

harm_batches = [token_batch(x) for x in harmful]
safe_batches = [token_batch(x) for x in safe]

harm_loader = DataLoader(harm_batches, batch_size=BATCH_SIZE)
safe_loader = DataLoader(safe_batches, batch_size=BATCH_SIZE)

# =====================================================================
# INTEGRATED GRADIENTS NEURON SCORING
# =====================================================================

# Ensure we have number of layers
num_layers = base_model.config.num_hidden_layers
hidden_size = base_model.config.hidden_size

# Prepare capture buffers & hooks
captures = [
    {"mlp": [], "qkv": [], "o": []}
    for _ in range(num_layers)
]

def process_attn_output(t):
    # t may be (B, seq, heads, head_dim) or (B, seq, hidden_size)
    if t.dim() == 4:
        # q,k,v → collapse to hidden size
        B, S, H, D = t.shape
        t = t.reshape(B, S, H * D)
    # Now sum over batch + sequence
    return t.abs().sum(dim=(0,1))

def make_ig_hook(layer_idx,module_type):
    def hook(module, inp, out):
        # detach to CPU to avoid holding computation graph / large GPU memory
        try:
            captures[layer_idx][module_type].append(out.detach().cpu())
        except Exception:
            # fallback: try converting to tensor then append
            try:
                captures[layer_idx][module_type].append(torch.tensor(out).detach().cpu())
            except Exception:
                # if even that fails, ignore
                pass
    return hook

# Register hooks (and keep handles to remove later)
ig_hooks = []
for i in range(num_layers):
    # MLP down_proj (Phi-3 style)
    try:
        mod_mlp = base_model.model.layers[i].mlp.down_proj
        ig_hooks.append(mod_mlp.register_forward_hook(make_ig_hook(i, "mlp")))
    except Exception:
        # some model variants may not have same attr; ignore
        pass
    try:
        attn = base_model.model.layers[i].self_attn
        # ig_hooks.append(attn.q_proj.register_forward_hook(make_ig_hook(i, "q")))
        # ig_hooks.append(attn.k_proj.register_forward_hook(make_ig_hook(i, "k")))
        # ig_hooks.append(attn.v_proj.register_forward_hook(make_ig_hook(i, "v")))
        ig_hooks.append(attn.qkv_proj.register_forward_hook(make_ig_hook(i, "qkv")))
        ig_hooks.append(attn.o_proj.register_forward_hook(make_ig_hook(i, "o")))
    except Exception:
        # if attention path differs, ignore; we only register what exists
        pass

def process_capture_list_to_per_neuron(cap_list, device):
    """
    cap_list: list of CPU tensors captured by hooks (shapes vary).
    Returns a tensor on 'device' with shape (hidden_size,) representing per-neuron
    aggregated absolute activations across batch and sequence dims.
    """
    if not cap_list:
        return None
    # concat along first dim
    try:
        cat = torch.cat(cap_list, dim=0)  # still CPU
    except Exception:
        # fallback: attempt stacking
        try:
            cat = torch.stack(cap_list, dim=0)
        except Exception:
            return None
    if cat.numel() == 0:
        return None
    # Move to device for potential faster reductions
    cat = cat.to(device, non_blocking=True)

    # cat shape possibilities:
    #  - (B_total, S, H)                    => hidden dim at -1
    #  - (B_total, S, heads, head_dim)     => needs reshape to (B_total, S, H=heads*head_dim)
    #  - (B_total, S, ...) but final dim equals hidden_size
    # We want per-neuron = sum(abs(...), dims=(0,1))
    if cat.dim() == 4:
        # (B, S, heads, head_dim) -> (B, S, H)
        B, S, h, d = cat.shape
        cat = cat.reshape(B, S, h * d)
    # If 3D and last dim == hidden_size, okay.
    if cat.dim() == 3:
        reduce_dims = (0, 1)
        per_neuron = cat.abs().sum(dim=reduce_dims)  # (H,)
    elif cat.dim() == 2:
        # maybe (B_total, H) — treat sequence collapsed already
        per_neuron = cat.abs().sum(dim=0)
    else:
        # Unexpected shape — try flattening trailing dims into hidden axis
        last = cat.reshape(cat.shape[0], -1, cat.shape[-1])
        per_neuron = last.abs().sum(dim=(0,1))
    # Ensure length matches hidden_size if possible; otherwise leave as-is
    return per_neuron


# Helper: run forward with given inputs_embeds and capture activations (hooks fill captures)
def forward_with_embeds(inputs_embeds, attention_mask):
    # Using inputs_embeds triggers same forward pass as input_ids
    return base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

# --- (HELPER FUNCTION FOR PROCESSING) ---
def process_capture_list(cap_list, device):
    """Concatenates, moves to device, and sums activations for a list of captures."""
    if not cap_list:
        return None
    try:
        acts_cpu = torch.cat(cap_list, dim=0) 
        if acts_cpu.numel() == 0:
            return None
        acts = acts_cpu.to(device, non_blocking=True)
        # Sum over batch and sequence length dimensions
        reduce_dims = tuple(range(acts.dim() - 1)) 
        per_neuron_score = acts.abs().sum(dim=reduce_dims) # (H,)
        return per_neuron_score
    except Exception as e:
        print(f"Warning: Failed to process capture list: {e}")
        return None

# IG calculation per batch
def integrated_gradients_batch(batch):
    """
    Compute IG-style approximate per-layer, per-neuron scores for one batch.
    batch: dict with input_ids, attention_mask (tensors on CPU)
    returns: dict layer_idx -> tensor (hidden_size,) on CPU
    """
    # move token ids to device for embedding lookup
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    # get embeddings and baseline
    embed_layer = base_model.get_input_embeddings()
    emb = embed_layer(input_ids)                      # (B, L, H)
    baseline = torch.zeros_like(emb, device=DEVICE)   # zero baseline

    # accumulator for this batch (on device)
    mlp_acc = {i: torch.zeros(hidden_size, device=DEVICE) for i in range(num_layers)}
    attn_o_acc = {i: torch.zeros(hidden_size, device=DEVICE) for i in range(num_layers)}
    attn_qkv_acc = {i: torch.zeros(hidden_size, device=DEVICE) for i in range(num_layers)}

    alphas = torch.linspace(0.0, 1.0, steps=IG_STEPS, device=DEVICE)
    for alpha in alphas:
        # clear capture lists (they're CPU lists)
        for i in range(num_layers):
            for k in captures[i]:
                captures[i][k].clear()

        emb_step = baseline + alpha * (emb - baseline)
        emb_step.requires_grad_(True)

        # forward: hooks will append to captures
        out = forward_with_embeds(emb_step, attention_mask)

        # choose a plausible scalar to keep autograd happy (not strictly needed here)
        logits = out.logits  # (B, L, V)
        scalar = logits[:, -1, :].mean()
        # compute grad w.r.t embeddings, allow_unused True
        try:
            _ = torch.autograd.grad(scalar, [emb_step], retain_graph=False, allow_unused=True)[0]
        except Exception:
            pass

        # process captured lists for each layer and add to accumulators
        for layer_idx in range(num_layers):
            # MLP down-proj
            mlp_per = process_capture_list_to_per_neuron(captures[layer_idx]["mlp"], DEVICE)
            if mlp_per is not None:
                mlp_acc[layer_idx] += mlp_per.detach()

            # Attention O-proj
            o_per = process_capture_list_to_per_neuron(captures[layer_idx]["o"], DEVICE)
            if o_per is not None:
                attn_o_acc[layer_idx] += o_per.detach()

            # Attention q/k/v aggregated:
            qkv_per = process_capture_list_to_per_neuron(captures[layer_idx]["qkv"], DEVICE)
            if qkv_per is not None:
                expected_shape_acc = attn_qkv_acc[layer_idx].shape[0]

                if qkv_per.shape[0] == expected_shape_acc * 3:
                    # Split into Q, K, V scores (each is [3072])
                    q_scores, k_scores, v_scores = torch.tensor_split(qkv_per, 3)
                    
                    # Sum them to get a single combined score vector of shape [3072]
                    qkv_total_score = q_scores + k_scores + v_scores
                    
                    # Add to the accumulator
                    attn_qkv_acc[layer_idx] += qkv_total_score.detach()
                    
                elif qkv_per.shape[0] == expected_shape_acc:
                    # Fallback in case shape is already correct
                    attn_qkv_acc[layer_idx] += qkv_per.detach()
                
                # else: shape mismatch, print warning or skip
                # else:
                #    print(f"Skipping L{layer_idx} QKV: shape {qkv_per.shape[0]} != {expected_shape_acc}")

        # free step tensors
        del emb_step, scalar, logits
        torch.cuda.empty_cache()

    # average across IG steps and move to CPU
    mlp_scores = {l: (mlp_acc[l] / float(IG_STEPS)).cpu() for l in mlp_acc}
    attn_o_scores = {l: (attn_o_acc[l] / float(IG_STEPS)).cpu() for l in attn_o_acc}
    attn_qkv_scores = {l: (attn_qkv_acc[l] / float(IG_STEPS)).cpu() for l in attn_qkv_acc}
    return mlp_scores, attn_o_scores, attn_qkv_scores

# ---------------------------
# Aggregate over dataloader
# ---------------------------
def collect_means(loader, tag):
    print(f"[IG] collecting means for {tag} ...")
    mlp_sum = {i: torch.zeros(hidden_size, device="cpu") for i in range(num_layers)}
    attn_o_sum = {i: torch.zeros(hidden_size, device="cpu") for i in range(num_layers)}
    attn_qkv_sum = {i: torch.zeros(hidden_size, device="cpu") for i in range(num_layers)}
    count = 0

    for batch in tqdm(loader, desc=f"Scoring {tag}"):
        mlp_scores, attn_o_scores, attn_qkv_scores = integrated_gradients_batch(batch)
        for l, vec in mlp_scores.items():
            mlp_sum[l] += vec.cpu()
        for l, vec in attn_o_scores.items():
            attn_o_sum[l] += vec.cpu()
        for l, vec in attn_qkv_scores.items():
            attn_qkv_sum[l] += vec.cpu()
        count += 1

    # means
    mlp_means = {l: (mlp_sum[l] / max(1, count)) for l in mlp_sum}
    attn_o_means = {l: (attn_o_sum[l] / max(1, count)) for l in attn_o_sum}
    attn_qkv_means = {l: (attn_qkv_sum[l] / max(1, count)) for l in attn_qkv_sum}

    # save for inspection
    torch.save(mlp_means, os.path.join(OUTPUT_DIR, f"{tag}_mlp_ig_means.pt"))
    torch.save(attn_o_means, os.path.join(OUTPUT_DIR, f"{tag}_attn_o_ig_means.pt"))
    torch.save(attn_qkv_means, os.path.join(OUTPUT_DIR, f"{tag}_attn_qkv_ig_means.pt"))

    return mlp_means, attn_o_means, attn_qkv_means

# ---------------------------
# Run scoring collection
# ---------------------------
harm_mlp_means, harm_attn_o_means, harm_attn_qkv_means = collect_means(harm_loader, "harmful")
safe_mlp_means, safe_attn_o_means, safe_attn_qkv_means = collect_means(safe_loader, "safe")

# remove hooks
for h in ig_hooks:
    try:
        h.remove()
    except Exception:
        pass

# ---------------------------
# Compute diffs & select top-K per module
# ---------------------------
def select_topk_from_means(harm_means, safe_means, topk):
    """Return list of (layer, neuron_idx) sorted by descending (harm - safe)."""
    scores = []
    for l in range(num_layers):
        # ensure both exist (if missing, treat as zeros)
        hm = harm_means.get(l)
        sm = safe_means.get(l)
        if hm is None: hm = torch.zeros(hidden_size)
        if sm is None: sm = torch.zeros(hidden_size)
        diff = hm - sm  # tensor (H,)
        # push tuples (score, layer, idx)
        for idx in range(diff.numel()):
            scores.append((float(diff[idx].item()), l, idx))
    scores.sort(key=lambda x: x[0], reverse=True)
    topk_list = [(l, idx) for _, l, idx in scores[:topk]]
    return topk_list

# MLP
# top_mlp_safety = select_topk_from_means(harm_mlp_means, safe_mlp_means, TOP_K_NEURONS)
# top_mlp_utility = select_topk_from_means(safe_mlp_means, harm_mlp_means, TOP_K_NEURONS)  # utility = safe - harm

# # Attn O-proj
# top_attn_o_safety = select_topk_from_means(harm_attn_o_means, safe_attn_o_means, TOP_K_NEURONS)
# top_attn_o_utility = select_topk_from_means(safe_attn_o_means, harm_attn_o_means, TOP_K_NEURONS)

# Attn QKV aggregated
top_attn_qkv_safety = select_topk_from_means(harm_attn_qkv_means, safe_attn_qkv_means, TOP_K_NEURONS)
top_attn_qkv_utility = select_topk_from_means(safe_attn_qkv_means, harm_attn_qkv_means, TOP_K_NEURONS)

# Save JSON indices
# json.dump(top_mlp_safety, open(os.path.join(OUTPUT_DIR, "mlp_safety_indices.json"), "w"))
# json.dump(top_mlp_utility, open(os.path.join(OUTPUT_DIR, "mlp_utility_indices.json"), "w"))
# json.dump(top_attn_o_safety, open(os.path.join(OUTPUT_DIR, "attn_o_safety_indices.json"), "w"))
# json.dump(top_attn_o_utility, open(os.path.join(OUTPUT_DIR, "attn_o_utility_indices.json"), "w"))
json.dump(top_attn_qkv_safety, open(os.path.join(OUTPUT_DIR, "attn_qkv_safety_indices.json"), "w"))
json.dump(top_attn_qkv_utility, open(os.path.join(OUTPUT_DIR, "attn_qkv_utility_indices.json"), "w"))

print("Saved index files to:", OUTPUT_DIR)
print("Done.")
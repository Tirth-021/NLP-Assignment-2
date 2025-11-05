# %% Cell 1: Imports and Setup
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from huggingface_hub import login
import copy
import os
import gc
import logging
import sys
import math
import random
from collections import defaultdict
from itertools import cycle, islice
import time
import json

# --- (NEW) Setup Logging ---
log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('neuron_tune.log')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# --- Configuration ---
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K_NEURONS = 1024  # Number of neurons to modulate
SAFE_REFUSAL_RESPONSE = "I'm sorry, I cannot assist with that request."

login(token='')

# %% Cell 2: Load Model and Tokenizer
clear_gpu_memory()
log.info(f"Loading model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
base_model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
log.info("Model and tokenizer loaded.")

# %% Cell 2.5: Load and Process Datasets (unchanged)
DEBUG_SAMPLE_SIZE = 200

def format_safety_data(example):
    return {
        "text_input": example['prompt'],
        "text_target": SAFE_REFUSAL_RESPONSE
    }

def format_utility_data(example):
    if example.get('input'):
        text_input = f"Instruction: {example['instruction']}\nInput: {example['input']}"
    else:
        text_input = f"Instruction: {example['instruction']}"
    return {
        "text_input": text_input,
        "text_target": example['output']
    }

safety_data = load_dataset("walledai/AdvBench", split="train")
safety_data = safety_data.map(format_safety_data)
safety_data_split = safety_data.train_test_split(test_size=0.5, seed=42)
D_safety_global = safety_data_split['train']
D_safety_local = safety_data_split['test']

utility_data = load_dataset("tatsu-lab/alpaca", split="train")
utility_data = utility_data.shuffle(seed=42).select(range(DEBUG_SAMPLE_SIZE))
utility_data = utility_data.map(format_utility_data)
utility_data_split = utility_data.train_test_split(test_size=0.5)
D_utility_global = utility_data_split['train']
D_utility_local = utility_data_split['test']

log.info(f"Safety data: {len(D_safety_global)} global, {len(D_safety_local)} local")
log.info(f"Utility data: {len(D_utility_global)} global, {len(D_utility_local)} local")

# %% Cell 2.6: PyTorch Dataset Class (unchanged)
class NeuronTuneDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        inputs = self.tokenizer(
            item['text_input'],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        targets = self.tokenizer(
            item['text_target'],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        target_ids = targets.input_ids.squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": target_ids
        }

train_safety_dataset = NeuronTuneDataset(D_safety_global, tokenizer)
val_safety_dataset = NeuronTuneDataset(D_safety_local, tokenizer)
train_utility_dataset = NeuronTuneDataset(D_utility_global, tokenizer)
val_utility_dataset = NeuronTuneDataset(D_utility_local, tokenizer)

def load_idx_file(name):
    path = os.path.join("./neuron_scores", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    arr = json.load(open(path, 'r'))
    # ensure tuples
    return [tuple(x) for x in arr]

mlp_safety_indices = load_idx_file("mlp_safety_indices.json")
mlp_utility_indices = load_idx_file("mlp_utility_indices.json")
attn_safety_indices = load_idx_file("attn_safety_indices.json")
attn_utility_indices = load_idx_file("attn_utility_indices.json")

log.info(f"Loaded index sizes: mlp_safety={len(mlp_safety_indices)}, mlp_utility={len(mlp_utility_indices)}, attn_safety={len(attn_safety_indices)}, attn_utility={len(attn_utility_indices)}")


# %% Cell 3: Step 1 - Pinpointing Neurons (UPDATED)

# Global dict to store activations and replacement dict for IG steps
mlp_activations = {}
_activation_replacements = {}  # layer_idx -> tensor (used during IG interpolation)

def get_mlp_hook(layer_idx):
    """
    pre-hook captures activation and supports replacement for integrated gradients.
    Also calls retain_grad when not replacing to ensure activation.grad is available for fallback.
    """
    def hook(module, input,output):
        act = input[0]
        # Store the live activation (not detached) so activation.grad can be filled if needed
        mlp_activations[layer_idx] = act
        # If a replacement tensor has been prepared for this layer, return it to substitute the activation
        rep = _activation_replacements.get(layer_idx, None)
        if rep is not None:
            # replace activation with replacement tensor (must match shape)
            return (rep,) + input[1:]
        # Ensure non-leaf activations retain grad if downstream code will read activation.grad
        # This is cheap relative to integrated gradients and necessary for fallback method.
        if not act.requires_grad:
            try:
                act.requires_grad_(True)
            except Exception:
                pass
        try:
            act.retain_grad()
        except Exception:
            # ignore if unavailable
            pass
        return None
    return hook

# Register hooks on mlp.down_proj modules (same as original)
hooks = []
# Note: Keep the same indexing scheme used originally
for i in range(base_model.config.num_hidden_layers):
    # Capture MLP down_proj activations
    try:
        mlp_proj = base_model.model.layers[i].mlp.down_proj
        h1 = mlp_proj.register_forward_hook(get_mlp_hook(i))   # <-- forward_hook (not pre)
        hooks.append(h1)
    except Exception:
        pass

    # Capture Attention output projection activations
    try:
        attn_proj = base_model.model.layers[i].self_attn.o_proj
        h2 = attn_proj.register_forward_hook(get_mlp_hook(i))  # <-- new hook
        hooks.append(h2)
    except Exception:
        pass

log.info(f"Registered {len(hooks)} hooks on MLP + Attention projections down_projs for attribution.")

# --- Helper: Representative attack selection (paper: one representative per attack class) ---
def select_representative_attacks(hf_dataset, max_per_class=1):
    """
    Group entries by available attack-class metadata if present.
    Fallback to grouping by prompt fingerprint (first 120 chars) if no metadata.
    """
    class_candidates = ['attack_class', 'attack_type', 'category', 'class', 'attack']
    groups = defaultdict(list)
    for ex in hf_dataset:
        cls = None
        for k in class_candidates:
            if k in ex and ex[k]:
                cls = ex[k]
                break
        if cls is None:
            # fallback: fingerprint on prompt prefix
            prompt = ex.get('prompt') or ex.get('text_input') or ''
            cls = (prompt[:120].strip() or 'unknown')
        groups[cls].append(ex)
    selected = []
    for cls, examples in groups.items():
        selected.extend(examples[:max_per_class])
    if not selected:
        # fallback: take first N
        selected = list(islice(hf_dataset, 0, 10))
    return selected

# --- Attribution helpers: integrated-gradients (paper Eq.4) and fallback ---
def _model_forward_logits(model, input_ids, attention_mask):
    # simple forward returning logits
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

def calculate_attribution_scores(model, input_ids, attention_mask, labels,
                                 use_integrated_gradients=True, ig_steps=12):
    """
    Primary attribution function:
     - By default attempt integrated-gradients along path between baseline and attacked activation.
     - Fallback to simple gradient-based attribution using activation.retain_grad().
    Preserves API: returns (neuron_scores_dict, loss_value)
    """
    global mlp_activations, _activation_replacements
    mlp_activations = {}
    _activation_replacements = {}

    device_for_inputs = input_ids.device

    # First compute forward and loss for the given batch (attacked)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    # If integrated gradients requested, attempt IG:
    if use_integrated_gradients:
        try:
            # We need a baseline (benign) input. The dataset doesn't provide benign q explicitly.
            # Paper expects q (benign) and (p,q) (attacked). We fallback to synthetic benign:
            # create a short benign prompt (a neutral instruction) as baseline.
            # Keep tokenization consistent.
            benign_prompt = "Please provide a helpful and non-harmful response."
            benign_tokens = tokenizer(benign_prompt, return_tensors="pt", truncation=True,
                                      padding='max_length', max_length=input_ids.shape[1]).to(device_for_inputs)

            # Capture baseline activations (v_q)
            mlp_activations.clear()
            _activation_replacements.clear()
            with torch.no_grad():
                _ = model(input_ids=benign_tokens['input_ids'], attention_mask=benign_tokens['attention_mask'])
                v_q = {l: act.detach().clone().to(device_for_inputs) for l, act in mlp_activations.items()}

            # Capture attacked activations (v_pq)
            mlp_activations.clear()
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                v_pq = {l: act.detach().clone().to(device_for_inputs) for l, act in mlp_activations.items()}

            common_layers = sorted(set(v_q.keys()).intersection(v_pq.keys()))
            if not common_layers:
                raise RuntimeError("No common MLP activation layers captured for IG.")

            # accumulate grads per layer (per neuron)
            accum_grads = {l: torch.zeros(v_q[l].shape[-1], device=device_for_inputs) for l in common_layers}
            delta = {l: (v_pq[l] - v_q[l]) for l in common_layers}

            # For each interpolation step, set replacements and compute gradient wrt replacements
            for step in range(1, ig_steps + 1):
                alpha = float(step) / ig_steps
                _activation_replacements.clear()
                # prepare replacement tensors with requires_grad
                for l in common_layers:
                    v_a = (v_q[l] + alpha * delta[l]).clone().requires_grad_(True)
                    _activation_replacements[l] = v_a

                # forward with replacements (hooks will replace activations accordingly)
                # model forward should produce logits that depend on replacements
                logits_step = model(input_ids=input_ids, attention_mask=attention_mask).logits

                # scalar loss for grad calculation
                loss_step = loss_fct(logits_step.view(-1, logits_step.size(-1)), labels.view(-1))
                # compute gradients wrt each replacement tensor
                grads = torch.autograd.grad(loss_step, [ _activation_replacements[l] for l in common_layers ],
                                            retain_graph=False, allow_unused=True)
                for l, g in zip(common_layers, grads):
                    if g is None:
                        continue
                    # reduce batch+seq dims -> per-neuron
                    g_sum = g.abs().sum(dim=tuple(range(g.dim() - 1))).detach()
                    accum_grads[l] += g_sum

                # clear replacements for next step (hook will see none and normal activations will be used)
                _activation_replacements.clear()

            # compute contribution scores: (delta_mag * avg_grad)
            contrib_scores = {}
            for l in common_layers:
                avg_grad = accum_grads[l] / ig_steps
                delta_mag = delta[l].abs().sum(dim=tuple(range(delta[l].dim() - 1)))
                contrib = (delta_mag * avg_grad).detach().cpu()
                contrib_scores[l] = contrib

            # Return contribs and the scalar loss on attacked input
            return contrib_scores, loss.item()

        except Exception as e:
            # If anything fails, fallback to simple gradient-based method
            log.warning(f"Integrated gradients failed ({e}). Falling back to simple grad attribution.")
            # continue to fallback implementation below

    # Fallback: simple gradient-based attribution (ensure activation.retain_grad used)
    mlp_activations = {}
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    loss_fb = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    model.zero_grad()
    loss_fb.backward()

    neuron_scores = {}
    for layer_idx, activation in mlp_activations.items():
        grad = activation.grad if hasattr(activation, 'grad') else None
        if grad is None:
            # warn (should be rare since hook calls retain_grad())
            log.warning(f"Activation.grad is None for layer {layer_idx} in fallback; skipping.")
            continue
        contribution = grad.abs().sum(dim=(0, 1)).detach().cpu()
        neuron_scores[layer_idx] = contribution

    return neuron_scores, loss_fb.item()

# The rest of the pinpointing code uses calculate_attribution_scores similarly but now can
# call select_representative_attacks first. We'll integrate into pinpoint_neurons below.

def pinpoint_neurons(model, safety_dataloader, utility_dataloader, k=TOP_K_NEURONS):
    """
    Pinpoint neurons using integrated gradients over representative attacks (paper).
    Keeps the same external behavior but adds attack-aware selection and IG.
    """
    log.info("Pinpointing neurons (attack-aware integrated gradients where possible)...")
    # Unfreeze parameters for attribution computation (we don't actually optimize them here)
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    # We'll select representative attack examples from safety dataset to compute attributions
    # Convert HF Dataset batches to raw examples for selection:
    # (safety_dataloader yields tokenized items; we pick from original D_safety_local)
    representatives_raw = select_representative_attacks(D_safety_local, max_per_class=1)
    log.info(f"Selected {len(representatives_raw)} representative attacks for attribution.")

    # Build per-layer accumulators
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    total_safety_scores = {i: torch.zeros(hidden_size).to(DEVICE) for i in range(num_layers)}
    total_utility_scores = {i: torch.zeros(hidden_size).to(DEVICE) for i in range(num_layers)}

    # Safety: For each representative, compute IG attribution
    from torch.utils.data import DataLoader as _DL  # reuse tokenizer mapping
    for rep in representatives_raw:
        # rep is an HF dataset example with 'text_input' (we created mapping)
        example = {"text_input": rep['prompt'], "text_target": SAFE_REFUSAL_RESPONSE}  # preserve prior mapping
        # Tokenize attacked input
        enc_att = tokenizer(example['text_input'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        enc_tgt = tokenizer(example['text_target'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        enc_att = {k: v.to(DEVICE) for k, v in enc_att.items()}
        labels = enc_tgt['input_ids'].to(DEVICE)
        # IG-based scores
        scores_s, _ = calculate_attribution_scores(model, enc_att['input_ids'], enc_att['attention_mask'], labels, use_integrated_gradients=True, ig_steps=10)
        # accumulate
        for l, v in scores_s.items():
            total_safety_scores[l] += v.to(DEVICE)

    # Utility: use a small random sample (we keep same style as before)
    # iterate a few batches from utility_dataloader
    utility_samples = 50
    u_iter = iter(utility_dataloader)
    for _ in range(min(utility_samples, len(utility_dataloader))):
        try:
            batch = next(u_iter)
        except StopIteration:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        scores_u, _ = calculate_attribution_scores(model, batch['input_ids'], batch['attention_mask'], batch['labels'], use_integrated_gradients=False)
        for l, v in scores_u.items():
            total_utility_scores[l] += v.to(DEVICE)

    # Flatten and pick top K per original behavior
    flat_safety = []
    flat_utility = []
    for i in range(num_layers):
        for j in range(hidden_size):
            flat_safety.append((total_safety_scores[i][j].item(), i, j))
            flat_utility.append((total_utility_scores[i][j].item(), i, j))
    flat_safety.sort(key=lambda x: x[0], reverse=True)
    flat_utility.sort(key=lambda x: x[0], reverse=True)
    safety_neuron_indices = [(l, n) for s, l, n in flat_safety[:k]]
    utility_neuron_indices = [(l, n) for s, l, n in flat_utility[:k]]

    # cleanup hooks (same as original)
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass
    del total_safety_scores, total_utility_scores, flat_safety, flat_utility
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"Identified {len(safety_neuron_indices)} safety and {len(utility_neuron_indices)} utility neurons (top-{k}).")
    return safety_neuron_indices, utility_neuron_indices

# %% Cell 4: Modulated Model (unchanged functionally but alpha clamps added)
modulation_handles = []

class ModulatedLLM(nn.Module):
    def __init__(self, base_model, mlp_safety, mlp_utility,attn_o_safety, attn_o_utility, 
                 attn_qkv_safety, attn_qkv_utility):
        super().__init__()
        self.model = base_model
        for param in self.model.parameters():
            param.requires_grad = False
        self.mlp_safety = mlp_safety
        self.mlp_utility = mlp_utility
        self.attn_o_safety = attn_o_safety
        self.attn_o_utility = attn_o_utility
        self.attn_qkv_safety = attn_qkv_safety
        self.attn_qkv_utility = attn_qkv_utility

        self.mlp_safety_alphas = nn.Parameter(torch.full((len(self.mlp_safety),), 1.5, dtype=torch.float32))
        self.mlp_utility_alphas = nn.Parameter(torch.full((len(self.mlp_utility),), 0.5, dtype=torch.float32))
        
        self.attn_o_safety_alphas = nn.Parameter(torch.full((len(self.attn_o_safety),), 1.5, dtype=torch.float32))
        self.attn_o_utility_alphas = nn.Parameter(torch.full((len(self.attn_o_utility),), 0.5, dtype=torch.float32))
        
        self.attn_qkv_safety_alphas = nn.Parameter(torch.full((len(self.attn_qkv_safety),), 1.5, dtype=torch.float32))
        self.attn_qkv_utility_alphas = nn.Parameter(torch.full((len(self.attn_qkv_utility),), 0.5, dtype=torch.float32))

        self.mlp_safety_map = defaultdict(lambda: (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
        self.mlp_utility_map = defaultdict(lambda: (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
        self.attn_o_safety_map = defaultdict(lambda: (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
        self.attn_o_utility_map = defaultdict(lambda: (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
        self.attn_qkv_safety_map = defaultdict(lambda: (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
        self.attn_qkv_utility_map = defaultdict(lambda: (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))

# --- Populate all 6 maps ---
        for pos, (l, n) in enumerate(self.mlp_safety):
            arr_idx, arr_pos = self.mlp_safety_map[l]
            arr_idx = torch.cat([arr_idx, torch.tensor([n], dtype=torch.long)]) if arr_idx.numel() else torch.tensor([n], dtype=torch.long)
            arr_pos = torch.cat([arr_pos, torch.tensor([pos], dtype=torch.long)]) if arr_pos.numel() else torch.tensor([pos], dtype=torch.long)
            self.mlp_safety_map[l] = (arr_idx, arr_pos)

        for pos, (l, n) in enumerate(self.mlp_utility):
            arr_idx, arr_pos = self.mlp_utility_map[l]
            arr_idx = torch.cat([arr_idx, torch.tensor([n], dtype=torch.long)]) if arr_idx.numel() else torch.tensor([n], dtype=torch.long)
            arr_pos = torch.cat([arr_pos, torch.tensor([pos], dtype=torch.long)]) if arr_pos.numel() else torch.tensor([pos], dtype=torch.long)
            self.mlp_utility_map[l] = (arr_idx, arr_pos)

        for pos, (l, n) in enumerate(self.attn_o_safety):
            arr_idx, arr_pos = self.attn_o_safety_map[l]
            arr_idx = torch.cat([arr_idx, torch.tensor([n], dtype=torch.long)]) if arr_idx.numel() else torch.tensor([n], dtype=torch.long)
            arr_pos = torch.cat([arr_pos, torch.tensor([pos], dtype=torch.long)]) if arr_pos.numel() else torch.tensor([pos], dtype=torch.long)
            self.attn_o_safety_map[l] = (arr_idx, arr_pos)

        for pos, (l, n) in enumerate(self.attn_o_utility):
            arr_idx, arr_pos = self.attn_o_utility_map[l]
            arr_idx = torch.cat([arr_idx, torch.tensor([n], dtype=torch.long)]) if arr_idx.numel() else torch.tensor([n], dtype=torch.long)
            arr_pos = torch.cat([arr_pos, torch.tensor([pos], dtype=torch.long)]) if arr_pos.numel() else torch.tensor([pos], dtype=torch.long)
            self.attn_o_utility_map[l] = (arr_idx, arr_pos)

        for pos, (l, n) in enumerate(self.attn_qkv_safety):
            arr_idx, arr_pos = self.attn_qkv_safety_map[l]
            arr_idx = torch.cat([arr_idx, torch.tensor([n], dtype=torch.long)]) if arr_idx.numel() else torch.tensor([n], dtype=torch.long)
            arr_pos = torch.cat([arr_pos, torch.tensor([pos], dtype=torch.long)]) if arr_pos.numel() else torch.tensor([pos], dtype=torch.long)
            self.attn_qkv_safety_map[l] = (arr_idx, arr_pos)

        for pos, (l, n) in enumerate(self.attn_qkv_utility):
            arr_idx, arr_pos = self.attn_qkv_utility_map[l]
            arr_idx = torch.cat([arr_idx, torch.tensor([n], dtype=torch.long)]) if arr_idx.numel() else torch.tensor([n], dtype=torch.long)
            arr_pos = torch.cat([arr_pos, torch.tensor([pos], dtype=torch.long)]) if arr_pos.numel() else torch.tensor([pos], dtype=torch.long)
            self.attn_qkv_utility_map[l] = (arr_idx, arr_pos)

    def modulation_hook(self, layer_idx, module_type):
        def hook(module, input, output):
            act = output
            device = act.device
            dtype = act.dtype

            is_qkv = (module_type == "attn_qkv")
            if is_qkv:
                # act shape is (B, S, 9216)
                hidden_dim = act.shape[-1] # 9216
                scaling = torch.ones(hidden_dim, device=device, dtype=dtype)
                
                # QKV safety
                idxs, pos = self.attn_qkv_safety_map.get(layer_idx, (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                if idxs.numel():
                    vals = self.attn_qkv_safety_alphas[pos.to(device)].to(device, dtype=dtype)
                    scaling.index_copy_(0, idxs.to(device), vals)
                # QKV utility
                idxs, pos = self.attn_qkv_utility_map.get(layer_idx, (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                if idxs.numel():
                    vals = self.attn_qkv_utility_alphas[pos.to(device)].to(device, dtype=dtype)
                    scaling.index_copy_(0, idxs.to(device), vals)
            
            else:
                # act shape is (B, S, 3072)
                hidden_dim = act.shape[-1] # 3072
                scaling = torch.ones(hidden_dim, device=device, dtype=dtype)

            # MLP maps
                if module_type == "mlp":
                    # safety
                    idxs, pos = self.mlp_safety_map.get(layer_idx, (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                    if idxs.numel():
                        idxs_dev = idxs.to(device)
                        pos_dev = pos.to(device)
                        # gather alphas
                        vals = self.mlp_safety_alphas[pos_dev].to(device, dtype=dtype)
                        scaling.index_copy_(0, idxs_dev, vals)
                    # utility
                    idxs, pos = self.mlp_utility_map.get(layer_idx, (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                    if idxs.numel():
                        idxs_dev = idxs.to(device)
                        pos_dev = pos.to(device)
                        vals = self.mlp_utility_alphas[pos_dev].to(device, dtype=dtype)
                        scaling.index_copy_(0, idxs_dev, vals)

                # Attention maps
                elif module_type == "attn_o":
                    idxs, pos = self.attn_o_safety_map.get(layer_idx, (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                    if idxs.numel():
                        idxs_dev = idxs.to(device)
                        pos_dev = pos.to(device)
                        vals = self.attn_o_safety_alphas[pos_dev].to(device, dtype=dtype)
                        scaling.index_copy_(0, idxs_dev, vals)
                    idxs, pos = self.attn_o_utility_map.get(layer_idx, (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                    if idxs.numel():
                        idxs_dev = idxs.to(device)
                        pos_dev = pos.to(device)
                        vals = self.attn_o_utility_alphas[pos_dev].to(device, dtype=dtype)
                        scaling.index_copy_(0, idxs_dev, vals)

            # elementwise scaling: broadcast to (B, L, H)
            mod = act * scaling
            return (mod,) if isinstance(output, tuple) else mod
        return hook


    def _register_hooks(self):
        global modulation_handles
        for h in modulation_handles:
            try: h.remove()
            except Exception: pass
        modulation_handles = []
        # register new forward hooks
        for layer_idx in range(self.model.config.num_hidden_layers):
            # MLP down_proj
            try:
                mlp_mod = self.model.model.layers[layer_idx].mlp.down_proj
                h = mlp_mod.register_forward_hook(self.modulation_hook(layer_idx, "mlp"))
                modulation_handles.append(h)
            except Exception:
                pass
            # Attention o_proj
            # 2. Attention O_proj
            try:
                attn_o_mod = self.model.model.layers[layer_idx].self_attn.o_proj
                h_o = attn_o_mod.register_forward_hook(self.modulation_hook(layer_idx, "attn_o"))
                modulation_handles.append(h_o)
            except Exception as e:
                log.warning(f"Failed to hook self_attn.o_proj on L{layer_idx}: {e}")
            # 3. Attention QKV_proj
            try:
                attn_qkv_mod = self.model.model.layers[layer_idx].self_attn.qkv_proj
                h_qkv = attn_qkv_mod.register_forward_hook(self.modulation_hook(layer_idx, "attn_qkv"))
                modulation_handles.append(h_qkv)
            except Exception as e:
                log.warning(f"Failed to hook self_attn.qkv_proj on L{layer_idx}: {e}")
           
    def forward(self, *args, **kwargs):
        self._register_hooks()
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self._register_hooks()
        return self.model.generate(*args, **kwargs)

    def parameters(self, recurse=True):
       return [
            self.mlp_safety_alphas,
            self.mlp_utility_alphas,
            self.attn_o_safety_alphas,
            self.attn_o_utility_alphas,
            self.attn_qkv_safety_alphas,
            self.attn_qkv_utility_alphas
       ]
    
    def state_dict(self):
        # Return only the alpha parameters' state
        return {
            'mlp_safety_alphas': self.mlp_safety_alphas,
            'mlp_utility_alphas': self.mlp_utility_alphas,
            'attn_o_safety_alphas': self.attn_o_safety_alphas,
            'attn_o_utility_alphas': self.attn_o_utility_alphas,
            'attn_qkv_safety_alphas': self.attn_qkv_safety_alphas,
            'attn_qkv_utility_alphas': self.attn_qkv_utility_alphas,
        }

    def load_state_dict(self, state_dict):
        # Load the alpha parameters' state
        self.mlp_safety_alphas.data.copy_(state_dict['mlp_safety_alphas'])
        self.mlp_utility_alphas.data.copy_(state_dict['mlp_utility_alphas'])
        self.attn_o_safety_alphas.data.copy_(state_dict['attn_o_safety_alphas'])
        self.attn_o_utility_alphas.data.copy_(state_dict['attn_o_utility_alphas'])
        self.attn_qkv_safety_alphas.data.copy_(state_dict['attn_qkv_safety_alphas'])
        self.attn_qkv_utility_alphas.data.copy_(state_dict['attn_qkv_utility_alphas'])

    def save_safe_model(self, path="./neuron_tune_artifact"):
        if not os.path.exists(path):
            os.makedirs(path)
        artifact = {
            'mlp_safety_indices': self.mlp_safety,
            'mlp_utility_indices': self.mlp_utility,
            'attn_o_safety_indices': self.attn_o_safety,
            'attn_o_utility_indices': self.attn_o_utility,
            'attn_qkv_safety_indices': self.attn_qkv_safety,
            'attn_qkv_utility_indices': self.attn_qkv_utility,
            'state_dict': self.state_dict() # Use the custom state_dict method
        }
        torch.save(artifact, os.path.join(path, "neuron_tune_artifact.pt"))
        tokenizer.save_pretrained(path)
        log.info(f"Saved NeuronTune artifact to {path}")

# %% Cell 5: MAML Training Loop (Second-Order MAML) with alpha clamping & diagnostics
def eval_loss(model, batch):
    model.eval()
    first_param_device = model.model.get_input_embeddings().weight.device
    batch = {k: v.to(first_param_device) for k, v in batch.items()}

    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
    return loss

def safe_mean(tensor):
    """Safely computes mean of a tensor, returning nan if empty."""
    if tensor.numel() == 0:
        return float('nan')
    return tensor.detach().cpu().float().mean().item()


def run_maml_training(modulated_model, global_safety_data, global_utility_data, local_safety_data, local_utility_data, epochs=3, K=5):
    log.info("Starting meta-learning (First-Order MAML) training...")
    # NOTE: We can turn this off for FO-MAML, it's less prone to graph errors
    # torch.autograd.set_detect_anomaly(True)
    # --- (NEW) FIX: Move meta-parameters to the correct device ---
    model_device = next(modulated_model.model.parameters()).device
    modulated_model.to(model_device) 
    # --- (END FIX) ---

    meta_optimizer = optim.Adam([
        {"params": modulated_model.mlp_safety_alphas, "lr": 1e-2},
        {"params": modulated_model.mlp_utility_alphas, "lr": 1e-3},
        {"params": modulated_model.attn_o_safety_alphas, "lr": 1e-2},
        {"params": modulated_model.attn_o_utility_alphas, "lr": 1e-3},
        {"params": modulated_model.attn_qkv_safety_alphas, "lr": 1e-2},
        {"params": modulated_model.attn_qkv_utility_alphas, "lr": 1e-3}
    ])
    lambda_val = 0.8

    g_safety_loader = DataLoader(global_safety_data, batch_size=2, shuffle=True)
    g_utility_loader = DataLoader(global_utility_data, batch_size=1, shuffle=True)
    l_safety_loader = DataLoader(local_safety_data, batch_size=2, shuffle=True)
    l_utility_loader = DataLoader(local_utility_data, batch_size=1, shuffle=True)

    l_safety_iter = cycle(l_safety_loader)
    l_utility_iter = cycle(l_utility_loader)

    for epoch in range(epochs):
        log.info(f"--- Epoch {epoch+1}/{epochs} ---")
        global_batches = iter(zip(g_safety_loader, g_utility_loader))
        for i, (b_s, b_u) in enumerate(global_batches):
            
            # --- (FIX 1: Create a new model with *cloned* leaf parameters) ---
            cloned = ModulatedLLM(modulated_model.model, 
                modulated_model.mlp_safety, modulated_model.mlp_utility,
                modulated_model.attn_o_safety, modulated_model.attn_o_utility,
                modulated_model.attn_qkv_safety, modulated_model.attn_qkv_utility)

            dev = next(modulated_model.model.parameters()).device
            # clone alphas as leaf parameters and ensure requires_grad True
            cloned.mlp_safety_alphas = nn.Parameter(modulated_model.mlp_safety_alphas.detach().clone().to(dev).requires_grad_(True))
            cloned.mlp_utility_alphas = nn.Parameter(modulated_model.mlp_utility_alphas.detach().clone().to(dev).requires_grad_(True))
            cloned.attn_o_safety_alphas = nn.Parameter(modulated_model.attn_o_safety_alphas.detach().clone().to(dev).requires_grad_(True))
            cloned.attn_o_utility_alphas = nn.Parameter(modulated_model.attn_o_utility_alphas.detach().clone().to(dev).requires_grad_(True))
            cloned.attn_qkv_safety_alphas = nn.Parameter(modulated_model.attn_qkv_safety_alphas.detach().clone().to(dev).requires_grad_(True))
            cloned.attn_qkv_utility_alphas = nn.Parameter(modulated_model.attn_qkv_utility_alphas.detach().clone().to(dev).requires_grad_(True))

            # copy the per-layer index maps (they are CPU tensors in your implementation)
            cloned.mlp_safety_map = {k: (v[0].clone(), v[1].clone()) for k, v in modulated_model.mlp_safety_map.items()}
            cloned.mlp_utility_map = {k: (v[0].clone(), v[1].clone()) for k, v in modulated_model.mlp_utility_map.items()}
            cloned.attn_o_safety_map = {k: (v[0].clone(), v[1].clone()) for k, v in modulated_model.attn_o_safety_map.items()}
            cloned.attn_o_utility_map = {k: (v[0].clone(), v[1].clone()) for k, v in modulated_model.attn_o_utility_map.items()}
            cloned.attn_qkv_safety_map = {k: (v[0].clone(), v[1].clone()) for k, v in modulated_model.attn_qkv_safety_map.items()}
            cloned.attn_qkv_utility_map = {k: (v[0].clone(), v[1].clone()) for k, v in modulated_model.attn_qkv_utility_map.items()}

            # === Crucial: re-register hooks so cloned's hook closures refer to cloned alphas ===
            cloned._register_hooks()
            
            # --- (FIX 2: Re-introduce the inner_optimizer) ---
            # This now works because the cloned parameters are leaf tensors.
            inner_optimizer = optim.Adam(cloned.parameters(), lr=1e-2)

            # --- (FIX 3: Use standard inner loop, no create_graph) ---
            for k_step in range(K):
               # Local safety update (minimize loss)
                d_local_s = next(l_safety_iter)
                inner_optimizer.zero_grad()
                l_s_local = eval_loss(cloned, d_local_s)
                l_s_local.backward()
                inner_optimizer.step()
                del l_s_local
                clear_gpu_memory()

                # Local utility update
                d_local_u = next(l_utility_iter)
                inner_optimizer.zero_grad()
                l_u_local = eval_loss(cloned, d_local_u)
                (-l_u_local).backward()
                inner_optimizer.step()
                del l_u_local
                clear_gpu_memory()

            # --- (End of inner loop) ---

            # Outer loop: evaluate on global batches
            l_s_global = eval_loss(cloned, b_s)
            l_u_global = eval_loss(cloned, b_u)
            l_joint = (lambda_val * l_s_global) + ((1 - lambda_val) * -l_u_global)
            
            meta_optimizer.zero_grad()
            # This calculates grads w.r.t. the *cloned* parameters
            l_joint.backward() 

            # --- (FIX 4: Manually copy gradients to meta-parameters) ---
            with torch.no_grad():
                if cloned.mlp_safety_alphas.grad is not None:
                    modulated_model.mlp_safety_alphas.grad = cloned.mlp_safety_alphas.grad.clone()
                if cloned.mlp_utility_alphas.grad is not None:
                    modulated_model.mlp_utility_alphas.grad = cloned.mlp_utility_alphas.grad.clone()
                
                if cloned.attn_o_safety_alphas.grad is not None:
                    modulated_model.attn_o_safety_alphas.grad = cloned.attn_o_safety_alphas.grad.clone()
                if cloned.attn_o_utility_alphas.grad is not None:
                    modulated_model.attn_o_utility_alphas.grad = cloned.attn_o_utility_alphas.grad.clone()

                if cloned.attn_qkv_safety_alphas.grad is not None:
                    modulated_model.attn_qkv_safety_alphas.grad = cloned.attn_qkv_safety_alphas.grad.clone()
                if cloned.attn_qkv_utility_alphas.grad is not None:
                    modulated_model.attn_qkv_utility_alphas.grad = cloned.attn_qkv_utility_alphas.grad.clone()
            # Meta-optimizer steps the *original* modulated_model parameters
            meta_optimizer.step()

            with torch.no_grad():
                modulated_model.mlp_safety_alphas.data.clamp_(1.0, 3.0)
                modulated_model.attn_o_safety_alphas.data.clamp_(1.0, 3.0)
                modulated_model.attn_qkv_safety_alphas.data.clamp_(1.0, 3.0)
                
                modulated_model.mlp_utility_alphas.data.clamp_(0.5, 1.0)
                modulated_model.attn_o_utility_alphas.data.clamp_(0.5, 1.0)
                modulated_model.attn_qkv_utility_alphas.data.clamp_(0.5, 1.0)

            # --- (Grad norm logging remains the same) ---
            with torch.no_grad():
                s_grad_norm = 0.0
                if modulated_model.mlp_safety_alphas.grad is not None:
                    s_grad_norm += modulated_model.mlp_safety_alphas.grad.norm().item()
                if modulated_model.attn_o_safety_alphas.grad is not None:
                    s_grad_norm += modulated_model.attn_o_safety_alphas.grad.norm().item()
                if modulated_model.attn_qkv_safety_alphas.grad is not None:
                    s_grad_norm += modulated_model.attn_qkv_safety_alphas.grad.norm().item()
                
                u_grad_norm = 0.0
                if modulated_model.mlp_utility_alphas.grad is not None:
                        u_grad_norm += modulated_model.mlp_utility_alphas.grad.norm().item()
                if modulated_model.attn_o_utility_alphas.grad is not None:
                    u_grad_norm += modulated_model.attn_o_utility_alphas.grad.norm().item()
                if modulated_model.attn_qkv_utility_alphas.grad is not None:
                    u_grad_norm += modulated_model.attn_qkv_utility_alphas.grad.norm().item()
                log.info(f"  Grad norms — safety: {s_grad_norm:.6f}, utility: {u_grad_norm:.6f}")


            if (i + 1) % 10 == 0:
                try:
                    # s_mlp_mean = modulated_model.mlp_safety_alphas.detach().cpu().float().mean().item()
                    # s_o_mean = modulated_model.attn_o_safety_alphas.detach().cpu().float().mean().item()
                    # s_qkv_mean = modulated_model.attn_qkv_safety_alphas.detach().cpu().float().mean().item()
                    
                    # u_mlp_mean = modulated_model.mlp_utility_alphas.detach().cpu().float().mean().item()
                    # u_o_mean = modulated_model.attn_o_utility_alphas.detach().cpu().float().mean().item()
                    # u_qkv_mean = modulated_model.attn_qkv_utility_alphas.detach().cpu().float().mean().item()

                    s_mlp_mean = safe_mean(modulated_model.mlp_safety_alphas)
                    s_o_mean = safe_mean(modulated_model.attn_o_safety_alphas)
                    s_qkv_mean = safe_mean(modulated_model.attn_qkv_safety_alphas)

                    u_mlp_mean = safe_mean(modulated_model.mlp_utility_alphas)
                    u_o_mean = safe_mean(modulated_model.attn_o_utility_alphas)
                    u_qkv_mean = safe_mean(modulated_model.attn_qkv_utility_alphas)

                    log.info(f"  Step {i+1} | Grad Norms (S/U): {s_grad_norm:.4f} / {u_grad_norm:.4f}")
                    log.info(f"  Means S (mlp/o/qkv): {s_mlp_mean:.4f} / {s_o_mean:.4f} / {s_qkv_mean:.4f}")
                    log.info(f"  Means U (mlp/o/qkv): {u_mlp_mean:.4f} / {u_o_mean:.4f} / {u_qkv_mean:.4f}")
                except Exception:
                    pass

            for p in modulated_model.parameters():
                try:
                    if p.grad is not None:
                        p.grad = None
                except Exception:
                    pass
            torch.cuda.empty_cache()
            gc.collect()
            del cloned, inner_optimizer, l_s_global, l_u_global, l_joint
            clear_gpu_memory()

            if (i+1) % 10 == 0:
                log.info(f"  Epoch {epoch+1}, Meta-Step {i+1} complete.")

        log.info(f"  Epoch {epoch+1} complete.")

    log.info("Meta-learning complete.")
    # torch.autograd.set_detect_anomaly(False) # No longer needed
    return modulated_model

# %% Cell 6: Main Execution (preserve original flow)
log.info("--- Starting NeuronTune Workflow ---")
attribution_safety_loader = DataLoader(val_safety_dataset, batch_size=1)
attribution_utility_loader = DataLoader(val_utility_dataset, batch_size=1)

# 1. Pinpoint Neurons
# safety_indices, utility_indices = pinpoint_neurons(
#     base_model,
#     attribution_safety_loader,
#     attribution_utility_loader,
#     k=TOP_K_NEURONS
# )
import json

# with open("/iitgn/home/tirth.bhatt/neuron_scores/mlp_safety_indices.json") as f:
#     mlp_safety_indices = [tuple(x) for x in json.load(f)]
# with open("/iitgn/home/tirth.bhatt/neuron_scores/mlp_utility_indices.json") as f:
#     mlp_utility_indices = [tuple(x) for x in json.load(f)]

# with open("/iitgn/home/tirth.bhatt/neuron_scores/attn_o_safety_indices.json") as f:
#     attn_o_safety_indices = [tuple(x) for x in json.load(f)]
# with open("/iitgn/home/tirth.bhatt/neuron_scores/attn_o_utility_indices.json") as f:
#     attn_o_utility_indices = [tuple(x) for x in json.load(f)]

mlp_safety_indices = []
mlp_utility_indices = []
attn_o_safety_indices = []
attn_o_utility_indices = []

with open("/iitgn/home/tirth.bhatt/neuron_scores/attn_qkv_safety_indices.json") as f:
    attn_qkv_safety_indices = [tuple(x) for x in json.load(f)]
with open("/iitgn/home/tirth.bhatt/neuron_scores/attn_qkv_utility_indices.json") as f:
    attn_qkv_utility_indices = [tuple(x) for x in json.load(f)]

log.info(f"MLP (S/U): {len(mlp_safety_indices)} / {len(mlp_utility_indices)}")
log.info(f"Attn-O (S/U): {len(attn_o_safety_indices)} / {len(attn_o_utility_indices)}")
log.info(f"Attn-QKV (S/U): {len(attn_qkv_safety_indices)} / {len(attn_qkv_utility_indices)}")


clear_gpu_memory()

# 2. Create Modulated Model
modulated_model = ModulatedLLM(base_model, 
    mlp_safety_indices, mlp_utility_indices,
    attn_o_safety_indices, attn_o_utility_indices,
    attn_qkv_safety_indices, attn_qkv_utility_indices)

# 3. Run MAML Training
log.info("Starting MAML training...")

safe_model = run_maml_training(
    modulated_model,
    train_safety_dataset, train_utility_dataset,
    val_safety_dataset, val_utility_dataset,
    epochs=10, K=5
)

# 4. Save the "Safe Model" Artifact
safe_model.save_safe_model("./my_neuron_tune_model")
log.info("\n--- NeuronTune Workflow Complete ---")

# %% Cell 7: How to Load and Use the Safe Model (unchanged behavior)
# log.info("\n--- Loading and Using the Safe Model ---")

# def load_safe_model(base_model, artifact_path):
#     artifact = torch.load(os.path.join(artifact_path, "neuron_tune_artifact.pt"))
#     safe_model = ModulatedLLM(base_model, artifact['safety_indices'], artifact['utility_indices'])
#     safe_model.load_state_dict(artifact['model_state_dict'])
#     safe_model.eval()
#     log.info("Safe model loaded with trained alpha factors.")
#     return safe_model

# clear_gpu_memory()

# base_model_for_inference = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     device_map=None
# )

# safe_model_inference = load_safe_model(base_model_for_inference, "./my_neuron_tune_model")

# log.info("\nRunning inference with the modulated safe model...")
# prompt = "Write a python script to reverse a string"
# input_device = safe_model_inference.model.get_input_embeddings().weight.device
# inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
# with torch.no_grad():
#     outputs = safe_model_inference.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

# log.info("Generated text:")
# log.info(tokenizer.decode(outputs[0], skip_special_tokens=True))

# compute_directions.py
import os
import torch
import torch.nn.functional as F
import numpy as np

SAE_DIR = "./sae_models"      # contains L{layer}_H{head}_sae.pt
ACT_DIR = "./acts_sae"        # contains L{layer}_H{head}_safe.npy and _harm.npy
OUT_DIR = "sae_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_HEADS = 32
HEAD_DIM = 3072 // NUM_HEADS

def load_sae_pt(path, device="cpu"):
    state = torch.load(path, map_location="cpu")
    # Expect 'encoder.weight' and 'decoder.weight'
    assert "encoder.weight" in state and "decoder.weight" in state, f"Missing keys in {path}"
    enc = state["encoder.weight"].clone()   # shape (latent, head_dim)
    dec = state["decoder.weight"].clone()   # shape (head_dim, latent)
    return enc, dec

DIRECTIONS = {}
skipped = 0
for fname in os.listdir(SAE_DIR):
    if not fname.endswith("_sae.pt"):
        continue
    # parse name L{layer}_H{head}_sae.pt
    try:
        parts = fname.replace(".pt","").split("_")
        layer = int(parts[0].replace("L",""))
        head = int(parts[1].replace("H",""))
    except Exception as e:
        print("Skipping unexpected file:", fname)
        continue

    safe_path = os.path.join(ACT_DIR, f"L{layer}_H{head}_safe.npy")
    harm_path = os.path.join(ACT_DIR, f"L{layer}_H{head}_harm.npy")
    sae_path = os.path.join(SAE_DIR, fname)
    if not (os.path.exists(safe_path) and os.path.exists(harm_path)):
        print(f"Warning: missing activations for L{layer} H{head}, skipping.")
        skipped += 1
        continue

    # load SAE
    E_cpu, D_cpu = load_sae_pt(sae_path, device="cpu")
    latent_dim, head_dim = E_cpu.shape
    assert head_dim == HEAD_DIM, f"Head dim mismatch {head_dim} != {HEAD_DIM} for {fname}"

    # load activation numpy arrays
    safe = np.load(safe_path)   # shape (N_safe, head_dim)
    harm = np.load(harm_path)   # shape (N_harm, head_dim)
    # convert to torch
    safe_t = torch.from_numpy(safe).float()   # (Nsafe, head_dim)
    harm_t = torch.from_numpy(harm).float()   # (Nharm, head_dim)

    # compute z = ReLU(x @ E.T)
    # E_cpu shape (latent, head_dim) -> E.T shape (head_dim, latent)
    E = E_cpu.float()
    z_safe = F.relu(safe_t @ E.t())   # (Nsafe, latent)
    z_harm = F.relu(harm_t @ E.t())   # (Nharm, latent)

    z_safe_mean = z_safe.mean(dim=0)
    z_harm_mean = z_harm.mean(dim=0)
    direction = z_harm_mean - z_safe_mean

    # normalize direction (unit vector). If direction is near-zero, skip
    norm = direction.norm().item()
    if norm < 1e-8:
        print(f"Direction near zero for L{layer} H{head} (norm {norm}), skipping.")
        skipped += 1
        continue
    #direction = direction / (norm + 1e-12)
    direction = direction.clone().half()
    DIRECTIONS[(layer, head)] = direction.clone()

print(f"Computed directions for {len(DIRECTIONS)} heads; skipped {skipped}.")

# Save directions (torch)
torch.save(DIRECTIONS, os.path.join(OUT_DIR, "directions.pt"))
print("Saved directions to:", os.path.join(OUT_DIR, "directions.pt"))

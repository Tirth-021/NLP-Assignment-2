import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# SAE architecture: z = ReLU(W_enc x), x' = W_dec z
class SAE(nn.Module):
    def __init__(self, d_model, d_latent):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_latent, bias=False)
        self.decoder = nn.Linear(d_latent, d_model, bias=False)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_recon = self.decoder(z)
        return x_recon, z

def train_sae_for_head(data, d_latent=256, lr=2e-4, steps=20000, device="cuda"):
    d_model = data.shape[-1]
    sae = SAE(d_model, d_latent).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    data = torch.tensor(data, dtype=torch.float32, device=device)

    for _ in tqdm(range(steps), desc="Training SAE"):
        idx = torch.randint(0, data.size(0), (4096,))
        batch = data[idx]

        rec, z = sae(batch)
        loss = ((rec - batch)**2).mean() + 1e-4 * z.abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return sae.cpu()

ACT_DIR = "acts_sae"
OUT_DIR = "sae_models"
os.makedirs(OUT_DIR, exist_ok=True)

# Infer layers & heads
files = [f for f in os.listdir(ACT_DIR) if "_safe.npy" in f]
for fname in files:
    layer = int(fname.split("_")[0][1:])
    head = int(fname.split("_")[1][1:])
    
    safe = np.load(f"{ACT_DIR}/L{layer}_H{head}_safe.npy")
    harm = np.load(f"{ACT_DIR}/L{layer}_H{head}_harm.npy")
    data = np.concatenate([safe, harm], axis=0)  # training set

    print(f"Training SAE for Layer {layer}, Head {head}, Data Shape = {data.shape}")
    sae = train_sae_for_head(data, d_latent=16)
    torch.save(sae.state_dict(), f"{OUT_DIR}/L{layer}_H{head}_sae.pt")

print("✅ All SAEs trained and saved.")

#!/bin/bash
#SBATCH --job-name=try5
#SBATCH --output=try5.out
#SBATCH --error=try5.err
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu          # ✅ change based on sinfo output
#SBATCH --gres=gpu:2            # ✅ adjust or remove based on scontrol show nodes

echo "[INFO] Job started on $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA info:"
nvidia-smi || echo "No GPU detected"

# Export Hugging Face and Perspective keys
export HF_TOKEN="XXX"
export PERSPECTIVE_API_KEY="XXX"

export USE_4BIT=1
export MAX_NEW_TOKENS=32          # start small; you can raise later
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


source activate sankaEnv
python try5.py

echo "[INFO] Job finished on $(date)"
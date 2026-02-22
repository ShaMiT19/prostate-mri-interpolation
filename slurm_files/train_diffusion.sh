#!/bin/bash
#SBATCH --job-name=train_diffusion
#SBATCH --output=logs/diffusion/train_diffusion_%j.out
#SBATCH --error=logs/diffusion/train_diffusion_%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

/orange/pinaki.sarder/s.savant/conda/envs/prostate_b200/bin/python scripts/train_diffusion.py

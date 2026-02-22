#!/bin/bash
#SBATCH --job-name=eval_diffusion
#SBATCH --output=logs/diffusion/eval_diffusion_%j.out
#SBATCH --error=logs/diffusion/eval_diffusion_%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

python scripts/eval_diffusion.py

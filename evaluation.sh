#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --output=logs/evaluation/eval_%j.out
#SBATCH --error=logs/evaluation/eval_%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Print job info
 /orange/pinaki.sarder/s.savant/conda/envs/prostate_b200/bin/python scripts/evaluation.py

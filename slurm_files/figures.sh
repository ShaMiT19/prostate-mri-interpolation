#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --output=paper_figures/logs/fig%j.out
#SBATCH --error=paper_figures/logs/fig%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Print job info
python scripts/generate_all_figures.py

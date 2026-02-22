#!/bin/bash
#SBATCH --job-name=train_cnn_upgraded
#SBATCH --output=logs/upgraded_cnn/train_cnn_upgraded_%j.out
#SBATCH --error=logs/upgraded_cnn/train_cnn_upgraded_%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

python scripts/train_cnn_upgraded.py

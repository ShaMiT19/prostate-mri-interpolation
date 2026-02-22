#!/bin/bash
#SBATCH --job-name=train_gan
#SBATCH --output=logs/gan/train_gan_%j.out
#SBATCH --error=logs/gan/train_gan_%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

 /orange/pinaki.sarder/s.savant/conda/envs/prostate_b200/bin/python scripts/train_gan.py

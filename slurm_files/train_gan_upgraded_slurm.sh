#!/bin/bash
#SBATCH --job-name=train_gan_upgraded
#SBATCH --output=logs/gan_upgraded/train_gan_upgraded_%j.out
#SBATCH --error=logs/gan_upgraded/train_gan_upgraded_%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

 /orange/pinaki.sarder/s.savant/conda/envs/prostate_b200/bin/python scripts/train_gan_upgraded.py

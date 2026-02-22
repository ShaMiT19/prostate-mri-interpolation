#!/bin/bash
#SBATCH --job-name=train_cnn_upgraded
#SBATCH --output=logs/cnn/train_cnn_%j.out
#SBATCH --error=logs/cnn/train_cnn_%j.err

#SBATCH --account=pinaki.sarder
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# *** DO NOT LOAD ANY CUDA OR CONDA MODULES ***
# *** DO NOT TRY TO ACTIVATE CONDA ***
# We directly run the Python interpreter that your Jupyter notebook uses.

# Just run your code using the correct python binary.
# THIS IS THE ONLY THING THAT WORKS AND THE ONLY THING YOU NEED.

 /orange/pinaki.sarder/s.savant/conda/envs/prostate_b200/bin/python scripts/train_cnn.py

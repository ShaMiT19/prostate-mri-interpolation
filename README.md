# Deep Learning for MRI Slice Interpolation

**Author:** Shamit Savant  
**MS Electrical & Computer Engineering – University of Florida**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project investigates deep learning approaches for MRI slice interpolation in prostate imaging. The objective is to reconstruct high-quality intermediate slices from sparsely sampled MRI volumes, improving volumetric continuity and anatomical detail.

Multiple model families were implemented, trained, and rigorously evaluated under a unified experimental pipeline:

- **CNN** (EDSR-style super-resolution)
- **CNN + U-Net hybrid**
- **Basic GAN**
- **Improved GAN with SE attention**
- **Diffusion-based model**

The project emphasizes reproducibility, structured experimentation, and quantitative evaluation.


---

## Publications

**[Main Report](./reports/ShamitSavant_DeepLearning_Main_Report_FinalProject.pdf)** | **[Supplementary Materials](./reports/ShamitSavant_DeepLearning_Supplementary_Report_FinalProject.pdf)**

> **TL;DR:** U-Net achieved **30.08 dB PSNR** and **0.898 SSIM**. Key finding: **Problem formulation has 230× more impact than architecture choice.**

---

---

## Problem Motivation

MRI scans often contain anisotropic resolution with large inter-slice gaps. Accurate interpolation of missing slices can:

- Improve 3D reconstruction quality
- Enhance downstream segmentation performance
- Reduce acquisition time
- Improve clinical visualization

This project evaluates modern deep learning methods to generate intermediate slices and compares their reconstruction fidelity.

---

## Models Implemented

### 1. CNN (EDSR-style)
- Residual super-resolution architecture
- L1 reconstruction loss
- Strong baseline PSNR performance

### 2. CNN + U-Net Hybrid
- Multi-scale encoder-decoder structure
- Enhanced structural consistency
- Improved SSIM performance

### 3. GAN (Basic)
- Generator + Discriminator framework
- Adversarial + reconstruction loss
- Improved perceptual realism

### 4. GAN with SE Attention
- Channel-wise attention mechanism
- Better texture preservation

### 5. Diffusion Model
- Iterative denoising-based reconstruction
- Strong structural continuity
- Higher computational cost

---

## Evaluation Metrics

Models were evaluated using:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- Error maps
- Side-by-side visual comparisons
- Radar plots and heatmaps

Both quantitative and qualitative evaluations were conducted across multiple experimental runs.

---

## Project Structure

```
├── checkpoints/          # Best trained model checkpoints
├── evaluation_outputs/   # Detailed metrics and sample outputs
├── final_evaluation/     # Aggregated experiment runs
├── logs/                 # Training logs
├── paper_figures/        # Figures used in final report
├── scripts/              # Training, evaluation, preprocessing code
├── train_*_slurm.sh      # HiPerGator SLURM job scripts
├── prostate_final.yml    # Conda environment file
├── prostate_sr.yml       # Alternate environment file
└── README.md             # This file
```

**Note:** Large raw datasets (~80GB) are not included in this repository.
I've included all the slurm files in the slurm_files folder for cleaner repository structure. Please make sure to place them in correct location or change the paths accordingly before running in your system. 

---

## ⚙️ Reproducibility

### 1. Environment Setup

```bash
conda env create -f prostate_final.yml
conda activate prostate_final
```

**Note** : Kept the yml files in "yml files" folder for cleaner repo structure.

### 2. Dataset

Place the dataset at:

```
/blue/username
```

Or update dataset paths inside `scripts/dataset.py`.

### 3. Training Example

**Local training:**
```bash
python scripts/train_cnn.py
```

**Note** : Make sure to include proper GPU access before running. 

**Or via SLURM (HiPerGator):**
```bash
sbatch train_cnn_slurm.sh
```

---

## Compute Environment

Experiments were conducted on:

- **University of Florida HiPerGator HPC cluster**
- Multi-GPU SLURM environment
- CUDA-enabled compute nodes

---

## Key Findings

- CNN-based models achieved strong PSNR performance
- GAN variants improved perceptual sharpness and texture realism
- Diffusion models produced structurally consistent outputs but required significantly more compute
- Attention mechanisms improved fine-detail reconstruction

---

## Future Work

- 3D volumetric modeling
- Transformer-based architectures
- Cross-dataset generalization
- Clinical validation and downstream segmentation evaluation

---

## Author

**Shamit Savant**  
MS Electrical & Computer Engineering  
University of Florida

For questions or collaboration inquiries, feel free to reach out.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Advisor:** Prof. Wei Shao, University of Florida
- **Computing Resources:** University of Florida HiPerGator Supercomputing Cluster
- **Dataset:** UCLA Prostate MRI-US Biopsy Dataset from The Cancer Imaging Archive (TCIA)

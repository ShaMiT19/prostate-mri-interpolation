Deep Learning for MRI Slice Interpolation
Final Project Submission - Folder Structure and Modifications
Author: Shamit Savant


1. Overview
-----------

This document explains the organization of the submitted project folder, the modifications made relative to the original working repository, and important notes for reproducibility. All essential components needed for grading, verification, and understanding the project have been included. Certain large or unnecessary items were intentionally removed to comply with submission size requirements and to keep the structure clean.


2. Removed Items and Rationale
------------------------------

The following items from the original working directory are not included in the submission:

(1) The data/ folder
    - This folder contains the processed prostate MRI dataset (~80 GB).
    - It is too large to include in a project submission.
    - The dataset was externally provided and is not required for grading.
    - All scripts will work once the dataset is restored at the expected path.

(2) Empty or unused directories
    - Folders generated during experimentation that do not contain meaningful content.

(3) Automatic cache or bytecode files
    - Python .pyc files
    - Temporary logs not relevant to the final results
    - These files do not contribute to reproducibility.

(4) Intermediate or duplicate outputs not used in the final report
    - Only essential evaluation outputs and figures were retained.


3. Contents Included in the Submission
--------------------------------------

The following materials remain and are required for complete understanding and evaluation of the project.

(1) Model Checkpoints (folder: checkpoints/)
    - CNN (EDSR-style) best checkpoint
    - CNN-Unet hybrid best checkpoint
    - Basic GAN (Generator + Discriminator)
    - Improved GAN with SE attention
    - Diffusion model best checkpoint

(2) Evaluation Outputs
    - k = 1 evaluation (run folders, comparison images, detailed metrics)
    - k = 2 evaluation metrics and sample visualizations
    - Diffusion model evaluation results, summary files, and visualizations

(3) Source Code (folder: scripts/)
    Includes all scripts required to train, evaluate, preprocess, and generate figures:
    - dataset.py
    - preprocess.py
    - train_cnn.py
    - train_cnn_upgraded.py
    - train_gan.py
    - train_gan_upgraded.py
    - diffusion_model.py
    - train_diffusion.py
    - eval_diffusion.py
    - evaluation.py
    - generate_all_figures.py
    These scripts are unchanged except for removal of large or unnecessary external files.

(4) Slurm Scripts
    - Training and evaluation job scripts used on HiPerGator (e.g., train_cnn_slurm.sh, train_gan_slurm.sh, evaluation.sh)
    - These document the computational environment and resource usage.

(5) Logs
    - Selected logs showing training progress and evaluation runs.
    - Included for transparency but trimmed to remove unnecessary bulk.

(6) Figures for the Final Report (folder: paper_figures/)
    - Side-by-side comparisons
    - Error maps
    - Zoomed visual patches
    - PSNR/SSIM scatter plot
    - Radar chart
    - SSIM heatmap
    These correspond directly to the figures referenced in the final report.

(7) Conda Environment Files
    - prostate_sr.yml
    - prostate_final.yml
    These files allow recreation of the exact Python environment used for the experiments.


4. Notes for Reproducibility
----------------------------

(1) Dataset Path
    Scripts expect the dataset at the following location (as used on HiPerGator):
        /blue/pinaki.sarder/s.savant/prostate_sr_project/data/raw/
    To rerun training or evaluation locally, place the dataset in a corresponding folder
    or update the dataset path inside the scripts.

(2) Evaluation Notes
    Also the naming convention is changed a bit according to the original working directory. I'm pasting the tree original tree strcuture below for reference. 
    Please don't hesitate to reach out to me in case you wish to run this on your system. I'll be more than happy to guide you run it on your system. 

├── checkpoints
│   ├── cnn
│   ├── cnn_1
│   ├── cnn_unet
│   ├── cnn_unet_1
│   ├── diffusion
│   ├── gan
│   ├── gan_1
│   ├── gan_improved
│   └── gan_improved_1
├── configs
├── data
│   ├── processed
│   ├── raw
│   ├── test
│   ├── train
│   └── val
├── eval_diffusion.sh
├── evaluation_outputs
│   ├── CNN_EDSR_detailed_results.txt
│   ├── CNN_EDSR_sample_0.png
│   ├── CNN_EDSR_sample_1.png
│   ├── CNN_EDSR_sample_2.png
│   ├── CNN_EDSR_sample_3.png
│   ├── CNN_EDSR_sample_4.png
│   ├── CNN_UNet_detailed_results.txt
│   ├── CNN_UNet_sample_0.png
│   ├── CNN_UNet_sample_1.png
│   ├── CNN_UNet_sample_2.png
│   ├── CNN_UNet_sample_3.png
│   ├── CNN_UNet_sample_4.png
│   ├── evaluation_summary.txt
│   ├── GAN_Basic_detailed_results.txt
│   ├── GAN_Basic_sample_0.png
│   ├── GAN_Basic_sample_1.png
│   ├── GAN_Basic_sample_2.png
│   ├── GAN_Basic_sample_3.png
│   ├── GAN_Basic_sample_4.png
│   ├── GAN_Improved_detailed_results.txt
│   ├── GAN_Improved_sample_0.png
│   ├── GAN_Improved_sample_1.png
│   ├── GAN_Improved_sample_2.png
│   ├── GAN_Improved_sample_3.png
│   └── GAN_Improved_sample_4.png
├── evaluation.sh
├── figures.sh
├── final_evaluation
│   ├── results_summary.txt
│   ├── Run1
│   ├── Run2
│   ├── Run3
│   ├── Run4
│   ├── Run5
│   ├── Run6
│   ├── Run7
│   ├── Run8
│   └── visualizations
├── logs
│   ├── baseline
│   ├── cnn
│   ├── cnn_1
│   ├── diffusion
│   ├── evaluation
│   ├── gan
│   ├── gan_1
│   ├── gan_upgraded
│   ├── gan_upgraded_1
│   ├── upgraded_cnn
│   └── upgraded_cnn_1
├── models
├── paper_figures
│   ├── Run1
│   └── Run2
├── prostate_final.yml
├── prostate_sr.yml
├── results
│   ├── cnn
│   ├── diffusion
│   ├── gan
│   └── visualizations
├── scripts
│   ├── calculate_baseline.py
│   ├── dataset.py
│   ├── diffusion_model_1.py
│   ├── diffusion_model_2.py
│   ├── diffusion_model_3.py
│   ├── diffusion_model.py
│   ├── eval_diffusion_1.py
│   ├── eval_diffusion_2.py
│   ├── eval_diffusion_3.py
│   ├── eval_diffusion_4.py
│   ├── eval_diffusion.py
│   ├── evaluation_1.py
│   ├── evaluation_2_working.py
│   ├── evaluation_3.py
│   ├── evaluation_deep_research.py
│   ├── evaluation_new.py
│   ├── evaluation.py
│   ├── generate_all_figures.py
│   ├── preprocess.py
│   ├── __pycache__
│   ├── train_cnn_1.py
│   ├── train_cnn.py
│   ├── train_cnn_upgraded_1.py
│   ├── train_cnn_upgraded_2.py
│   ├── train_cnn_upgraded.py
│   ├── train_diffusion_1.py
│   ├── train_diffusion_2.py
│   ├── train_diffusion.py
│   ├── train_gan_1.py
│   ├── train_gan.py
│   ├── train_gan_upgraded_1.py
│   ├── train_gan_upgraded_2.py
│   └── train_gan_upgraded.py
├── train_cnn_slurm.sh
├── train_cnn_upgraded_slurm.sh
├── train_diffusion.sh
├── train_gan_slurm.sh
├── train_gan_upgraded_slurm.sh
└── utils


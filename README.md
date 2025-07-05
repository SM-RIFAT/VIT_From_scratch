# Vision Transformer for Image Classification

## Project Overview
This repository contains a Python implementation of a Vision Transformer (ViT) for image classification, specifically applied to a dataset of pizza, steak, and sushi images. The project utilizes PyTorch and torchvision to fine-tune a pretrained `ViT_B_16` model from torchvision's model zoo for classifying images into three categories: pizza, steak, and sushi. The codebase, implemented in a Jupyter notebook (`ViT.ipynb`), includes data preprocessing, model setup, and preparation for training, leveraging transfer learning to adapt the ViT model for a custom dataset.

## Features
- **Dataset**: Uses a dataset of pizza, steak, and sushi images (10% and 20% subsets) downloaded from GitHub, split into training and test sets.
- **Model**: Fine-tunes a pretrained `ViT_B_16` model with a custom classification head for three classes, freezing the backbone to optimize training.
- **Data Preprocessing**: Applies image resizing (224x224) and normalization using torchvision transforms, tailored to ViT requirements.
- **Training Setup**: Configures data loaders with a batch size of 32 and prepares the model for training on GPU (CUDA) or CPU.
- **Utilities**: Incorporates helper functions for data downloading, seed setting, and loss curve plotting, with model summary visualization via `torchinfo`.
- **Environment**: Uses PyTorch 2.1.0 and torchvision 0.16.0, with automatic installation of required versions if needed.

## Repository Contents
- **Notebook**:
  - `ViT.ipynb`: Main notebook containing code for data downloading, preprocessing, model setup, and fine-tuning preparation.
- **Data**:
  - `pizza_steak_sushi/`: Directory for 10% dataset (train/test splits).
  - `pizza_steak_sushi_20_percent/`: Directory for 20% dataset (train split).
- **Dependencies**:
  - Python 3.11.5, PyTorch 2.1.0, torchvision 0.16.0, torchinfo, matplotlib.
  - External scripts from `pytorch-deep-learning` (e.g., `going_modular`, `helper_functions`).

## Requirements
- **Python Libraries**:
  ```bash
  pip install torch==2.1.0 torchvision==0.16.0 torchinfo matplotlib

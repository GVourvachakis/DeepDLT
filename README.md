# DLIP Data Analysis and (Variational) Autoencoder Showcase Project 

This repository contains a showcase project developed for the team of Prof. Pantazis at the Institute of Applied and Computational Mathematics (FORTH). 
The project focuses on exploring and evaluating Autoencoders (AE) and Variational Autoencoders (VAE) for unsupervised representation learning and image reconstruction tasks. 
It incorporates various techniques such as data augmentation, k-fold cross-validation, and model evaluation metrics to build and improve deep learning models for image processing tasks.

## Project Overview

**Project Duration:** This project was scheduled and completed within **5 days**. The entire pipeline, including dataset creation, model training, and evaluation, was designed to be a fast-paced, iterative process aimed at demonstrating the potential of Autoencoders and VAEs in a short time frame.

In this project, I develop an end-to-end pipeline for training and evaluating Autoencoders and VAEs. The pipeline includes:

- **Preprocessing**: Data augmentation (random cropping, brightness variation) and splitting datasets into training, validation, and test sets.

- **Model Architectures**: Implementation of Convolutional Autoencoders and Variational Autoencoders with detailed encoder-decoder architectures and latent space modeling.

- **Training and Evaluation**: Use of reconstruction loss, PSNR, SSIM, and KL divergence to evaluate model performance.

- **Generative Capabilities**: The VAE model demonstrates the ability to generate new samples using latent space manipulation.

## Report

A detailed report on the methodology, experiments, and results of this project can be found in the **DeepDLT.pdf** document. The report covers:

- The pipeline and architecture choices.
- Data preprocessing, augmentation, and model evaluation strategies.
- Results from the Autoencoder and VAE experiments, including analysis of performance metrics and challenges faced.

You can access the report here:  **[DeepDLT.pdf](./DeepDLT.pdf)**

## Project Structure

The repository is organized into several key components, each with a specific role in the pipeline:

### 1. `create_dataset.py`
This script creates a dataset directory (`./dataset`) containing cropped and brightness-varying images (64x64 resolution, BMP format for storage efficiency). It also generates CSV files with labeled datasets split into training, validation, and testing sets, sampled from the `./images/all_images.xlsx` file.

### 2. `dataset_loader.py`
Defines a custom dataset class `LaserDataset` (derived from PyTorch's `Dataset`). This class handles data loading, applies ordinal encoding for the "PP1" categorical feature, and prepares train/validation/test dataloaders.

### 3. `stratified_split.py`
Contains functions for performing **k-fold label-wise cross-validation**:
- `k_fold_cross_validation()`: Executes k-fold cross-validation (default `k=5`).
- `main_cross_validation()`: Manages the multilabel cross-validation scheme.

### 4. `training_pipeline.py`
Implements a comprehensive training pipeline for the autoencoder model, which includes monitoring:
- Basic loss values (MSE).
- Image quality metrics (PSNR, SSIM).
It saves the best-performing models based on these metrics.

### 5. `label_training.py`
Systematically evaluates model performance for different input features using k-fold cross-validation. It tracks performance metrics and provides visual feedback for each fold while maintaining separate results for each feature type.

### 6. `inference.py`
Provides functions for model inference, including visual comparisons and performance analysis through both numerical metrics and visual feedback. It also allows exploration of the VAE's generative capabilities (under development).

### 7. `train_vae.py`
Contains the script for training a **Variational Autoencoder (VAE)**. This script iteratively processes the data through training and validation phases and saves model checkpoints, tracking the best-performing model based on validation loss.

### 8. `environment.yml`
A YAML file containing the environment dependencies and requirements for setting up the project.

### 9. `main.ipynb`
A Jupyter notebook where the entire training and inference pipeline is implemented, providing a comprehensive walkthrough of the project.

### 10. `KFold_split.py` (not used)
A script for creating DataLoader instances for 5-fold cross-validation, which is not used in the current implementation.

Visit the DeepDLT.pdf

## Installation

To set up the project environment, use the provided `environment.yml` file to install dependencies. The easiest way to do this is using `conda`:

```bash
conda env create -f environment.yml
conda activate <your-environment-name>

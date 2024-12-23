#!/bin/bash

# Create a new Conda environment
conda create -n biobert_lora_env python=3.8 -y

# Activate the environment
source activate biobert_lora_env

# Install PyTorch (GPU or CPU-based)
# GPU installation:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# CPU installation:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install Hugging Face Transformers
pip install transformers

# Install PEFT (LoRA)
pip install peft

# Install Datasets for NLP data processing
pip install datasets

# Install YAML parsing library
pip install pyyaml

# Install tqdm for progress bars
pip install tqdm

# Install yacs for configuration management
pip install yacs

# Install scikit-learn for evaluation metrics
pip install scikit-learn

# Install tensorboard
pip install tensorboard

echo "Environment setup complete. Use the following command to activate the environment:"
echo "conda activate biobert_lora_env"

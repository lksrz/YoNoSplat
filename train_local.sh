#!/bin/bash

# Configuration for local YoNoSplat training on WinGPU WSL
# This script uses the shoes dataset rendered via Blender

export CUDA_HOME="/home/lukas/miniconda/envs/yonosplat"
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=gcc-11
export TORCH_CUDA_ARCH_LIST=8.9

# Source conda and activate env
source /home/lukas/miniconda/etc/profile.d/conda.sh
conda activate yonosplat

# Set working directory to repo root
cd "$(dirname "$0")"

# Run training
# Overrides:
# - dataset=shoes
# - trainer.devices=1 (use single GPU)
# - train.batch_size=1 (adjust based on VRAM)
# - mode=train

python src/main.py \
    dataset=shoes \
    trainer.devices=1 \
    train.batch_size=1 \
    mode=train \
    dataset.roots=["/mnt/c/Users/lukas/shoe_renders_final"] \
    $@

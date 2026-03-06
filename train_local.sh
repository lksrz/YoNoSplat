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

# Add current directory to PYTHONPATH so 'src' can be found
export PYTHONPATH=.

# Run the same experiment recipe as Modal.
# With 2 context views, max_img_per_gpu=2 keeps the mixed sampler at batch size 1.
python -m src.main \
    +experiment=shoes_224_finetune \
    dataset.shoes.roots='["/mnt/c/Users/lukas/shoe_renders_final"]' \
    dataset.shoes.view_sampler.max_img_per_gpu=2 \
    data_loader.train.num_workers=2 \
    checkpointing.save_weights_only=false \
    $@

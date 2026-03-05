# Research: YoNoSplat for Shoe Reconstruction

## Resources
- **GitHub:** https://github.com/cvg/yonosplat
- **White Paper:** https://arxiv.org/html/2511.07321v1
- **Project Page:** https://botaoye.github.io/yonosplat/

## Goal
3D Gaussian Splatting reconstruction from sparse, unposed images (approx. 10 views) for footwear. Leveraging Modal.com for CUDA-accelerated inference and potential future fine-tuning on proprietary 3D datasets.

## Implementation Details
- **Platform:** Modal.com (profile: `wearfits`)
- **Environment:** CUDA 11.8, PyTorch 2.1.2, custom Gaussian rasterizer.
- **Script:** `modal_app.py` in this directory.

## Testing Data
- Green Puma Clyde (9 images with markers) stored in `test_images/`.

---

## WSL Local Setup (verified — RTX 40xx, sm_89)

### 1) Create env

```bash
conda create -n yonosplat python=3.10 -y
conda activate yonosplat
```

### 2) Packaging tools (required for extension builds)

```bash
python -m pip install -U pip wheel
python -m pip install --force-reinstall "setuptools==69.5.1" packaging
```

### 3) PyTorch + CUDA 12.1 (aligned with nvcc 12.1)

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 4) CUDA compiler toolchain via conda

```bash
conda install -y -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev
```

### 5) Build env vars (per session)

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=gcc-11
export TORCH_CUDA_ARCH_LIST=8.9
```

### Install rasterizer

```bash
cd /home/lukasz/rasterizer
python -m pip install --no-build-isolation -e .
```

### Verify

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
python -c "import diff_gaussian_rasterization as dgr; print('dgr ok')"
```

### Common pitfalls

- Use `python -m pip` instead of `pip` (avoids wrong interpreter)
- `CUDA_HOME` must point to `$CONDA_PREFIX`, not a system path
- If `pkg_resources` error returns after updates: reinstall `setuptools==69.5.1`

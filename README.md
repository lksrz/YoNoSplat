# YoNoSplat — Shoe Fine-tuning

Fork of [cvg/YoNoSplat](https://github.com/cvg/YoNoSplat) for fine-tuning on shoe dataset.

## Resources
- **Original repo:** https://github.com/cvg/YoNoSplat
- **Paper:** https://arxiv.org/html/2511.07321v1
- **Project Page:** https://botaoye.github.io/yonosplat/

## Goal
Fine-tune YoNoSplat (3D Gaussian Splatting from sparse unposed images) on multi-view shoe renders for footwear reconstruction.

---

## Cloud Training (Modal.com)

### Setup (one-time)

1. **Modal account**: workspace `wearfits` at https://modal.com/wearfits
2. **Install Modal CLI**: `pip install modal`
3. **Auth**: `modal setup` or copy `~/.modal.toml` with tokens:
   ```toml
   [wearfits]
   token_id = "ak-..."
   token_secret = "as-..."
   active = true
   ```
4. **Volumes** (created automatically on first run, or manually):
   ```bash
   modal volume create yonosplat-dataset
   modal volume create yonosplat-checkpoints
   ```

### Upload dataset
```bash
# From machine with shoe renders:
modal volume put yonosplat-dataset /path/to/shoe_renders_final/ /

# Incremental update (same command — overwrites existing files):
modal volume put yonosplat-dataset /path/to/shoe_renders_final/ /

# Check what's uploaded:
modal volume ls yonosplat-dataset /
```

### Run training
```bash
# Full training (A100-80GB, batch=4, 50k steps)
MODAL_PROFILE=wearfits modal run modal_train.py --batch-size 4 --max-steps 50000

# Check dataset only
MODAL_PROFILE=wearfits modal run modal_train.py --check-only

# Resume from checkpoint
MODAL_PROFILE=wearfits modal run modal_train.py --batch-size 4 --resume-from /checkpoints/outputs/last.ckpt
```

### Download checkpoints
```bash
modal volume get yonosplat-checkpoints /checkpoints ./local_checkpoints
```

### Stop a running job
```bash
# List running apps
modal app list

# Stop by app ID
modal app stop ap-XXXXX
```

### Modal image details
- **Base**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`
- **GPU**: A100-80GB (CUDA arch 8.0)
- **Key packages**: PyTorch 2.4, Lightning 2.1.2, gsplat, diff-gaussian-rasterization (compiled on A100)
- **Timeout**: 24h max per run

### Cost estimate
- A100-80GB: ~$3.73/h
- 50k steps at batch=4, ~0.32 it/s → ~40h → ~$150
- Spot/preemptible not configured (add `cloud="oci"` for cheaper)

---

## Dataset Generation (Blender)

Shoe GLB models → multi-view renders using Blender Cycles.

### Script
```bash
# GPU rendering (fast, ~1s/view on RTX 4080)
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background \
  --python dataset-gen/blender_render_random.py -- \
  --output C:/Users/lukas/shoe_renders_final

# CPU rendering (slower, use when GPU needed for training)
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background \
  --python dataset-gen/blender_render_random_cpu.py -- \
  --output C:/Users/lukas/shoe_renders_final
```

### Output structure
```
shoe_renders_final/
├── shoe_id_1/
│   ├── view_000.png   # 512×512 RGBA
│   ├── view_001.png
│   ├── ...
│   ├── view_049.png
│   └── poses.json     # camera poses + intrinsics (written last)
├── shoe_id_2/
│   └── ...
```

- 50 views per shoe, 512×512, 64 Cycles samples
- `poses.json` written after all views → incomplete shoes (no poses.json) are skipped by the dataset loader
- GLB source: `C:\Users\lukas\Downloads\` (662 total)

---

## Local Training (WinGPU / WSL)

### Prerequisites
- RTX 4080 16GB (tight — batch=1 only, ~16GB VRAM at peak)
- WSL2 Ubuntu, Miniconda at `/home/lukas/miniconda/`
- Conda env `yonosplat` (Python 3.10)

### Run
```bash
cd /home/lukas/YoNoSplat
conda activate yonosplat
PYTHONPATH=/home/lukas/YoNoSplat python3 -u src/main.py --config-name main_smoke
```

### Config: `config/main_smoke.yaml`
- `batch_size: 1` (VRAM constraint on 16GB)
- `num_context_views: 2`, `num_target_views: 1`
- `input_image_shape: [518, 518]` (DINOv2 patch_size=14 requirement)
- `max_steps: 50000`
- `checkpointing.every_n_train_steps: 500`
- `wandb.mode: disabled`

### WSL env setup
```bash
conda create -n yonosplat python=3.10 -y
conda activate yonosplat
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
conda install -y -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev

# Build vars (per session)
export CUDA_HOME="$CONDA_PREFIX"
export CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=gcc-11
export TORCH_CUDA_ARCH_LIST=8.9

# Install rasterizer
cd ~/rasterizer
git submodule update --init --recursive
pip install --no-build-isolation -e .

# Verify
python -c "import diff_gaussian_rasterization as dgr; print('ok')"
```

---

## Key Fixes & Lessons

- **OOM on 16GB**: Sample only `num_target_views` (1) instead of all remaining views
- **MixedBatchSampler**: `__getitem__` must handle `(idx, num_context)` tuples
- **DINOv2 patch_size=14**: Images must be divisible by 14 → resize 512→518 in loader
- **Incomplete renders crash**: Dataset loader now checks `poses.json` exists before including a shoe
- **ModelCheckpoint**: Use `save_top_k=-1` (keep all) when `monitor=None`
- **PyTorch version**: Code requires ≥2.2 (`torch.nn.attention.SDPBackend`)
- **tee kills progress bar**: Use `script -c '...' logfile` instead of `| tee` to preserve TTY

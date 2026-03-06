# YoNoSplat - Shoe Fine-tuning

Fork of [cvg/YoNoSplat](https://github.com/cvg/YoNoSplat) adapted for multi-view shoe reconstruction from Blender renders.

## Resources
- Original repo: https://github.com/cvg/YoNoSplat
- Paper: https://arxiv.org/abs/2511.07321
- Project page: https://botaoye.github.io/yonosplat/
- Pretrained YoNoSplat weights: https://huggingface.co/botaoye/YoNoSplat
- Pi3 backbone weights: https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors

## What Changed In This Fork
- Added a custom `shoes` dataset loader for Blender renders with per-view poses and intrinsics.
- Replaced index-based sampling with a geometry-aware shoe view sampler.
- Added a dedicated `shoes_224_finetune` experiment for 224x224 fine-tuning.
- Switched training checkpoints to full Lightning state for exact resume.
- Added alpha-aware training for RGBA renders:
  - random background per training view,
  - fixed background for validation and inference,
  - foreground-only supervision for `mse`, `lpips`, and `perceptual` losses.
- Fixed `.ply` export so Gaussian viewers such as SuperSplat receive world-space orientation and viewer-friendly opacity values.

## Current Recommended Workflow
1. Upload a complete RGBA shoe dataset to the Modal dataset volume.
2. Run a short pilot with `shoes_224_finetune`.
3. Inspect novel-view renders and the exported `.ply`.
4. If quality is sane, continue to the full run.

If a checkpoint was trained before the alpha-aware background fix, do not resume it. Start a new run from the pretrained base checkpoint instead.

## Cloud Training (Modal.com)

### One-time setup
```bash
pip install modal
modal setup
```

Expected workspace/profile:
- Modal workspace: `wearfits`
- CLI profile: `wearfits`

Create the volumes once:
```bash
modal volume create yonosplat-dataset
modal volume create yonosplat-checkpoints
modal volume create yonosplat-weights
```

### Upload dataset
```bash
MODAL_PROFILE=wearfits modal volume put yonosplat-dataset /path/to/shoe_renders_final/ /
MODAL_PROFILE=wearfits modal volume ls yonosplat-dataset /
```

### Short pilot run
```bash
MODAL_PROFILE=wearfits modal run modal_train.py \
  --experiment shoes_224_finetune \
  --batch-size 4 \
  --num-workers 4 \
  --num-context-views 2 \
  --num-target-views 1 \
  --max-steps 5000 \
  --finetune dl3dv
```

### Full run
```bash
MODAL_PROFILE=wearfits modal run modal_train.py \
  --experiment shoes_224_finetune \
  --batch-size 4 \
  --num-workers 4 \
  --num-context-views 2 \
  --num-target-views 1 \
  --max-steps 50000 \
  --finetune dl3dv
```

### Exact resume
```bash
MODAL_PROFILE=wearfits modal run modal_train.py \
  --experiment shoes_224_finetune \
  --batch-size 4 \
  --num-workers 4 \
  --num-context-views 2 \
  --num-target-views 1 \
  --resume-from /checkpoints/outputs/last.ckpt \
  --max-steps 50000
```

### Dataset-only check
```bash
MODAL_PROFILE=wearfits modal run modal_train.py --check-only
```

### Download checkpoints
```bash
MODAL_PROFILE=wearfits modal volume get yonosplat-checkpoints /checkpoints ./local_checkpoints
```

### Stop a running job
```bash
modal app list
modal app stop ap-XXXXX
```

### Modal training notes
- Default training recipe: `config/experiment/shoes_224_finetune.yaml`
- Default pretrained initialization: `dl3dv.ckpt`
- Checkpoints are saved every `500` train steps.
- `--resume-from` maps to Lightning checkpoint resume, not encoder-only weight loading.
- `batch-size` is translated into the mixed sampler budget via `dataset.shoes.view_sampler.max_img_per_gpu`.

### Modal runtime reference
Observed on March 6, 2026 for the current shoe dataset and A100-80GB pilot runs:
- dataset split: `629 train / 33 val`
- early smoke test throughput: about `0.57 it/s`
- warmed-up `5k` run throughput: about `0.96-0.98 it/s`

Treat these as rough references, not guaranteed throughput.

## Inference And `.ply` Export

### Modal inference on a shoe from the dataset volume
```bash
MODAL_PROFILE=wearfits modal run modal_infer.py \
  --ckpt-path /checkpoints/outputs/last.ckpt \
  --shoe-name YOUR_SHOE_ID \
  --num-context-views 2 \
  --num-novel-views 4
```

Outputs are written to:
```text
/checkpoints/infer_outputs/YOUR_SHOE_ID/
```

Notes:
- `modal_infer.py` is a qualitative sanity-check path.
- It currently uses the first `N` views of the shoe as context views.
- Novel views are rendered on a simple orbit around the estimated scene center.
- If `--export-ply` is left enabled, the same run exports `shoe.ply`.

### Local custom inference without poses
```bash
python infer_custom.py \
  --images /path/to/images \
  --checkpoint /path/to/checkpoint.ckpt \
  --output /tmp/yonosplat_out
```

This is a no-pose utility for quick experiments. It is not the main shoe evaluation path.

### `.ply` export note
Older `.ply` files generated before the export fix may look corrupted in SuperSplat or similar viewers. Regenerate them with the current code.

## Dataset Generation (Blender)

Shoe GLB models are rendered into multi-view RGBA images with per-view camera poses and intrinsics.

### Example render command
```bash
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background \
  --python dataset-gen/blender_render_random.py -- \
  --output C:/Users/lukas/shoe_renders_final
```

### Output structure
```text
shoe_renders_final/
├── shoe_id_1/
│   ├── view_000.png
│   ├── view_001.png
│   ├── ...
│   ├── view_049.png
│   └── poses.json
├── shoe_id_2/
│   └── ...
```

Expected properties:
- RGBA PNG renders are preferred.
- `poses.json` is written last; directories without it are skipped.
- Default source resolution is `512x512`.
- The default assumption is `50` views per shoe, but the loader only requires enough valid views for the configured sampler.

### Camera and alpha handling
- `transform_matrix` is expected in Blender `c2w` convention.
- The loader converts Blender camera axes to OpenCV convention by flipping `Y` and `Z`.
- Intrinsics can be stored either in source-image pixel units or already normalized.
- Training uses random background compositing per view when alpha is available.
- Validation and inference use a fixed white background.
- Supervision uses the alpha mask so the model is not trained to reproduce the synthetic background.

## Local Training (WSL / WinGPU)

### Environment assumptions
- WSL2 Ubuntu
- Conda env named `yonosplat`
- CUDA toolchain exposed through the conda env

### Run
```bash
cd /home/lukas/YoNoSplat
conda activate yonosplat
./train_local.sh
```

### Local script behavior
`train_local.sh` runs the same `shoes_224_finetune` recipe as Modal and keeps `dataset.shoes.view_sampler.max_img_per_gpu=2` so a 16GB GPU can stay near batch size `1`.

### Minimal environment bootstrap
```bash
conda create -n yonosplat python=3.10 -y
conda activate yonosplat
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
conda install -y -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev

export CUDA_HOME="$CONDA_PREFIX"
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=gcc-11
export TORCH_CUDA_ARCH_LIST=8.9
```

## `shoes_224_finetune` Summary

Main config: `config/experiment/shoes_224_finetune.yaml`

Defaults:
- input resolution: `224x224`
- source resolution: `512x512`
- context / target views: `2 / 1`
- pose-free encoder training: enabled
- validation split: `5%`
- checkpoint interval: `500` steps
- checkpoint format: full-state Lightning checkpoint
- default losses: `mse`, `lpips`, `opacity`, `intrinsic`, `pose`

Sampler defaults:
- context separation: `20-60` degrees
- fallback context separation: `10-90` degrees
- target angle window: `5-25` degrees

## Key Fixes In This Fork
- Camera intrinsics are normalized in source-image coordinates before the shared crop/resize shim is applied.
- Shoe sampling is geometry-aware instead of frame-index-based.
- Relative-pose normalization matches the stock dataset behavior when enabled.
- Modal training uses the real shoe fine-tune recipe instead of the smoke config.
- Resume semantics use full training checkpoints.
- RGBA datasets no longer teach the model to reconstruct the synthetic background.
- PLY export matches the Gaussian geometry used by the renderer closely enough for external Gaussian viewers.

## Smoke-tested Status
Validated on March 6, 2026:
- 100-step Modal smoke test completed cleanly.
- 5k-step Modal pilot completed cleanly.
- Training loss decreased during both pilots.
- Validation ran successfully at step `5000`.
- Qualitative renders showed shoe silhouette learning.
- Viewer compatibility issues in exported `.ply` were fixed by the export update.

## Related Docs
- [DATASETS.md](DATASETS.md)
- [EVALUATION.md](EVALUATION.md)

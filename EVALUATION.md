# Evaluation

## Shoes (qualitative evaluation)

For the shoe fine-tuning workflow, the primary evaluation path is qualitative inference on a held-out shoe plus optional `.ply` export.

### Modal inference

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

This inference path:
- loads the checkpoint by `state_dict`, so it works with the current full-state training checkpoints,
- uses the first `N` shoe views as context,
- renders a small orbit of novel views,
- optionally exports `shoe.ply` for Gaussian viewers.

### Local no-pose inference

```bash
python infer_custom.py \
  --images /path/to/images \
  --checkpoint /path/to/checkpoint.ckpt \
  --output /tmp/yonosplat_out
```

This path is useful for sanity checks, but it is not equivalent to the posed shoe dataset path.

### Background and metrics note

For RGBA shoe data:
- training uses random per-view backgrounds plus alpha-masked supervision,
- validation and inference use a fixed white background for stable visual comparisons.

If you compare outputs across checkpoints, keep the evaluation background fixed.

### `.ply` export note

The current exporter derives orientation and scale from world-space covariance and writes viewer-friendly opacity values. If an older `.ply` looked corrupted in SuperSplat, regenerate it with the current code.

## Novel View Synthesis

The model supports multiple evaluation modes:

**Ground-truth pose, ground-truth intrinsics, 6 context views:**
```bash
python -m src.main \
  +experiment=yono_dl3dv \
  mode=test \
  wandb.name=dl3dv_6v_GTPoseGTIntrin \
  dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation \
  dataset.dl3dv.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json \
  checkpointing.load=pretrained_weights/dl3dv_224x224_ctx2to32.ckpt \
  test.save_image=true \
  dataset.dl3dv.view_sampler.num_context_views=6 \
  test.align_pose=false \
  model.encoder.pose_free=false \
  model.decoder.prune_opacity_threshold=0.005
```

**Predicted pose, ground-truth intrinsics, 6 context views:**
```bash
python -m src.main \
  +experiment=yono_dl3dv \
  mode=test \
  wandb.name=dl3dv_6v_PredPoseGTIntrin \
  dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation \
  dataset.dl3dv.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json \
  checkpointing.load=pretrained_weights/dl3dv_224x224_ctx2to32.ckpt \
  test.save_image=true \
  dataset.dl3dv.view_sampler.num_context_views=6 \
  test.align_pose=true \
  model.encoder.pose_free=true \
  model.decoder.prune_opacity_threshold=0.005
```

**Predicted pose, predicted intrinsics, 6 context views:**
```bash
python -m src.main \
  +experiment=yono_dl3dv \
  mode=test \
  wandb.name=dl3dv_6v_PredPosePredIntrin \
  dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation \
  dataset.dl3dv.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json \
  checkpointing.load=pretrained_weights/dl3dv_224x224_ctx2to32.ckpt \
  test.save_image=true \
  dataset.dl3dv.view_sampler.num_context_views=6 \
  test.align_pose=true \
  model.encoder.pose_free=true \
  model.encoder.backbone.use_pred_intrinsics_for_embed=true \
  model.decoder.prune_opacity_threshold=0.005
```

**Predicted pose, predicted intrinsics, 6 context views, with post-optimization:**
```bash
python -m src.main \
  +experiment=yono_dl3dv \
  mode=test \
  wandb.name=dl3dv_6v_PredPosePredIntrin_Opt \
  dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation \
  dataset.dl3dv.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json \
  checkpointing.load=pretrained_weights/dl3dv_224x224_ctx2to32.ckpt \
  test.save_image=true \
  dataset.dl3dv.view_sampler.num_context_views=6 \
  test.align_pose=true \
  model.encoder.pose_free=true \
  model.encoder.backbone.use_pred_intrinsics_for_embed=true \
  model.decoder.prune_opacity_threshold=0.005 \
  test.post_opt_gs=true \
  test.post_opt_gs_iter=200
```

## Pose Estimation
To evaluate camera pose estimation accuracy:
```bash
python -m src.eval_pose \
  +experiment=yono_dl3dv \
  +evaluation=eval_pose \
  dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation \
  dataset.dl3dv.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json \
  checkpointing.load=pretrained_weights/dl3dv_224x224_ctx2to32.ckpt \
  dataset.dl3dv.view_sampler.num_context_views=6
```

## Metrics Calculation
Compute metrics from saved predictions:
```bash
JAXTYPING_DISABLE=1 python -m src.scripts.compute_metrics \
  +experiment=yono_dl3dv \
  +evaluation=dl3dv_6v \
  dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation \
  dataset.dl3dv.view_sampler.num_context_views=6 \
  evaluation.side_by_side_path=./outputs/comparisons/dl3dv_6v
```

# Evaluation

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

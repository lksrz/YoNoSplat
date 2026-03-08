# YoNoSplat Training Chronicle

Training history and observations log for shoe Gaussian splatting model.

---

## Run 1 — First Full Training (2026-03-07 → 2026-03-08)

**Config:** `shoes_224_finetune_v1.yaml`
**Pretrained from:** DL3DV checkpoint (`dl3dv_224x224_ctx2to32.ckpt`)
**Dataset:** ~629 train / 33 val shoes, ~50 views each, 512×512 RGBA

### Parameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 (backbone: 1e-6) |
| Max steps | 50,000 |
| Batch size | 4 |
| Context views | 2 |
| Target views | 1 |
| Image resolution | 224×224 |
| pose_norm_method | `"none"` (scale=1.0, raw Blender coords) |
| relative_pose | `false` |
| augment | `false` |
| Silhouette loss weight | 0.1 |
| Opacity loss weight | 0.05 |
| LPIPS weight | 0.05 |
| MSE weight | 1.0 |
| Pose loss weight | 0.1 |
| Intrinsic loss weight | 0.5 |
| bounds_radius (static) | 1.5 |
| bounds_radius (dynamic) | max camera distance |
| prune_opacity_threshold | 0.03 |
| GT pose sampling ratio | 0.9 (decay steps 5k→15k) |
| Inference pose mode | **predicted only** (no GT option) |

### Training Metrics (final steps ~49800-50000)
- Loss range: 0.008 – 0.025
- Throughput: ~0.98 it/s on A100-80GB
- Completed epoch 318 at step 50000
- Training ran ~14h

### Observations — FAILED
- **Quality:** Essentially non-functional. Output is a gray/white sphere with barely visible shoe blob.
- **Zoomed out:** Massive semi-transparent gray sphere dominates the scene in SuperSplat.
- **Zoomed in:** Blurry, colorless shoe shape with no texture or detail. Everything white/gray.
- **No color:** Model outputs near-zero SH coefficients, no meaningful color information.

### Root Cause Analysis
1. **Coordinate frame mismatch at inference:** Encoder uses predicted poses (eval mode always uses predicted), but novel cameras are in GT frame. Predicted poses are normalized relative to view0 (identity), GT cameras are in absolute Blender frame → total misalignment.
2. **Learning rate too low for domain shift:** DL3DV pretrained at LR=2e-4, fine-tuned at LR=1e-5 (20× lower). With backbone LR=1e-6, features barely adapt to shoe domain in 50k steps.
3. **No pose normalization:** DL3DV uses `max_pairwise_d` (normalizes camera baseline to ~1). Shoes use `"none"` (raw Blender coords, cameras at ~2-5 units). Model's depth predictions are calibrated for DL3DV scale → scale mismatch.
4. **No relative pose:** DL3DV uses `relative_pose: true`. Shoes use `false`. Predicted poses are always view0-relative, but GT poses are absolute → frame inconsistency during training (10% predicted pose steps produce garbage).
5. **Only 2 context views:** DL3DV uses 6. Far too little geometric information for reliable 3D reconstruction.
6. **Weak anti-fog losses:** Silhouette=0.1, opacity=0.05 insufficient to suppress the fog/sphere artifact.

---

## Run 2 — Aggressive Fixes (planned 2026-03-08)

**Config:** `shoes_224_finetune_v2.yaml`
**Pretrained from:** DL3DV checkpoint (same)
**Goal:** Drastic parameter changes to see if direction changes.

### Key Changes from v1
| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| Learning rate | 1e-5 | **2e-4** | Match DL3DV pretrained LR |
| Context views | 2 | **6** | Match DL3DV, we have 50 views per shoe |
| pose_norm_method | `"none"` | **`"max_pairwise_d"`** | Match DL3DV scale expectations |
| relative_pose | `false` | **`true`** | Match DL3DV, fix GT/predicted pose frame mismatch |
| Silhouette loss | 0.1 | **1.0** | Aggressively suppress fog |
| Opacity loss | 0.05 | **0.2** | Aggressively suppress fog |
| Max steps | 50,000 | **150,000** | Match DL3DV training length |
| prune_opacity_threshold | 0.03 | **0.1** | Remove more fog from PLY export |
| GT poses at inference | no flag | **`--use-gt-poses` flag** | Fix coordinate frame mismatch |
| GT pose decay start | 5,000 | **80,000** | Match DL3DV (let model learn with GT longer) |
| GT pose decay end | 15,000 | **100,000** | Match DL3DV |
| Target views | 1 | **1** | Keep same |
| warm_up_steps | 1,000 | **1,000** | Keep same |

### Why `relative_pose: true` (was `false` in v1)

v1 used `relative_pose: false` reasoning that shoes are model-centric (object at origin).
This was wrong for two reasons:

1. **DL3DV pretrained model was trained with `relative_pose: true`** (base_dataset.yaml
   default). The backbone, depth predictions, and pose decoder all learned to work in a
   "first camera = origin" coordinate system. Keeping `false` means we're feeding the
   pretrained model data in a coordinate system it never saw.

2. **Frame mismatch between GT and predicted poses.** The encoder's pose decoder always
   normalizes predicted poses relative to view0 (`se3_inverse` in encoder line 228-229).
   With `relative_pose: false`, GT poses are in absolute Blender coords but predicted
   poses are in view0-relative coords. The pose loss then compares two incompatible
   frames → it can never converge properly. With `relative_pose: true`, both GT and
   predicted are in the same view0-relative frame → pose loss is meaningful.

The shoe object is still reconstructed correctly — it just lives in view0's coordinate
frame instead of world origin. Inference orbit cameras compute `scene_centre` from
context camera positions, so this works regardless of which frame we're in.

### Expected Impact
- Scale normalization + relative pose should fix coordinate frame issues
- Higher LR should enable actual learning of shoe features
- 6 context views provides much more 3D information
- Strong anti-fog losses should eliminate gray sphere
- GT pose flag at inference removes pose prediction uncertainty

---

## Run 2b — From Scratch with Pi3 Backbone (parallel, 2026-03-08)

**Config:** `shoes_224_scratch.yaml`
**Pretrained from:** Pi3 backbone only (ViT features, no 3D heads)
**Goal:** No DL3DV domain bias. Model-centric from the start. Compare with Run 2.

### Rationale
Run 2 adapts a DL3DV model (trained on indoor scenes) to shoes — requires matching
DL3DV's coordinate conventions (relative_pose, max_pairwise_d). Run 2b starts fresh
with only Pi3 ViT backbone weights, so we can use model-centric conventions that are
more natural for object reconstruction:

- `relative_pose: false` — object stays at world origin (no view0 shift)
- `pose_norm_method: "none"` — raw Blender coordinates (scale=1.0)
- No DL3DV assumptions baked into the heads

Risk: 629 shoes (~31k images) may be too small to train 3D heads from scratch.
Pi3 backbone provides good visual features but point/gaussian/camera decoders start random.

### Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | **1e-4** | Higher than v2 (from scratch needs more) |
| Backbone LR | 1e-5 | |
| Warm-up steps | 2000 | Longer warmup for stability |
| Context views | 6 | |
| Batch size | 1 | |
| pose_norm_method | `"none"` | Model-centric, raw Blender coords |
| relative_pose | `false` | Object at origin |
| Silhouette loss | 1.0 | Same anti-fog as v2 |
| Opacity loss | 0.2 | Same anti-fog as v2 |
| Max steps | 150,000 | |
| Pretrained | Pi3 backbone only | `--finetune pi3` |

### Comparison Plan
Check both runs at ~20-30k steps to see which direction is better.
If v2 (DL3DV finetune) converges faster, continue with that approach.
If scratch produces cleaner geometry, the domain shift was the core problem.

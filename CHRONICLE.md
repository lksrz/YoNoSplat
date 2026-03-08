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
| pose_norm_method | `"max_pairwise_d"` (inherited from shoes base config) |
| relative_pose | `true` (inherited from shoes base config) |
| augment | `true` (inherited from shoes base config) |
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
1. **Coordinate frame mismatch at inference:** Encoder uses predicted poses (eval mode always uses predicted), but novel cameras are in GT frame. Predicted poses are normalized relative to view0 (identity), GT cameras are in absolute Blender frame → total misalignment. (**Fixed in v2 with `--use-gt-poses` flag.**)
2. **Learning rate possibly too low for domain shift:** DL3DV pretrained at LR=2e-4, fine-tuned at LR=1e-5 (20× lower). However, Run 1 was perfectly stable at 1e-5 — the loss converged smoothly to 0.013. The issue was inference quality, not training convergence.
3. ~~**No pose normalization / No relative pose:**~~ **CORRECTION:** Run 1 actually DID use `relative_pose: true` and `pose_norm_method: max_pairwise_d` (inherited from shoes base dataset config). These were NOT the problem.
4. **Only 2 context views:** DL3DV uses 6. Far too little geometric information for reliable 3D reconstruction.
5. **Weak anti-fog losses:** Silhouette=0.1, opacity=0.05 may have been insufficient to suppress the fog/sphere artifact at inference.

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

---

## Run 2 & 2b Results — Both FAILED (2026-03-08)

### What Happened

Both runs were killed by the **loss filter** (`train_ignore_large_loss_mse`) after step 20,000.

The filter thresholds inherited from `main.yaml` defaults were calibrated for DL3DV/RE10K:
- `train_ignore_large_loss_mse: 0.06` — skips samples where MSE loss > 0.06
- `train_ignore_large_loss: 0.2` — skips samples where total loss > 0.2
- `train_ignore_large_loss_pose: 1.0`

For shoes, typical MSE loss is 0.08–0.20 (much higher than DL3DV's 0.01–0.05).
After step 20k, the filter skipped **every single sample** → model stopped learning.

### Run 2 (finetune v2) — Additional Issue: Gradient Explosion

LR=2e-4 was too aggressive for finetuning. Loss trajectory:
- Steps 0–6870: oscillating loss (0.03–0.28) with frequent spikes to ~15.5 but recovering
- **Step 6880: permanent collapse** — loss locked at ~15.5, never recovered
- Steps 7000–20000: loss ~15.5 constantly (model producing garbage)
- Steps 20000+: filter kicked in, skipping everything → completely dead

Spike frequency by 1k-step range: 0k=6, 1k=7, 2k=4, 3k=1, 4k=5, 5k=12, 6k=14, 7k+=100% spikes

### Run 2b (scratch) — Was Training Well

Loss trajectory (avg per 1k steps):
- 0k: 2.40, 1k: 0.68, 5k: 0.26, 10k: 0.19, 15k: 0.15, 19k: **0.14** ← healthy convergence
- **Step 20000: filter kicks in** → avg jumps to 2.9, then 9.7, then 12+ (all skipped)

No spikes > 1.0 after initial convergence. This was a healthy training killed by misconfigured filter.

### Saved Checkpoints

Both runs have checkpoints every 500 steps on Modal volume `yonosplat-checkpoints`:
- **Finetune v2:** `/outputs/exp_shoes_224_finetune_v2/2026-03-08_09-34-46/checkpoints/`
  - Best before collapse: `epoch=10-step=6500.ckpt`
- **Scratch v1:** `/outputs/exp_shoes_224_scratch/2026-03-08_09-39-44/checkpoints/`
  - Best (last healthy): `epoch=31-step=19500.ckpt`

---

## Run 3 — Resume with Fixed Filters (2026-03-08)

### Run 3a: Finetune v3 (resume from v2 step 6500)

**Config:** `shoes_224_finetune.yaml` (v3)
**Resume from:** v2 checkpoint `epoch=10-step=6500.ckpt`

Key changes from v2:
| Parameter | v2 | v3 | Rationale |
|-----------|----|----|-----------|
| Learning rate | 2e-4 | **5e-5** | Prevent gradient explosion |
| train_ignore_large_loss | 0.2 (default) | **5.0** | Only filter genuine explosions |
| train_ignore_large_loss_mse | 0.06 (default) | **1.0** | Shoe MSE is 0.08–0.20 normally |
| train_ignore_large_loss_pose | 1.0 (default) | **5.0** | Only filter explosions |

### Run 3b: Scratch v2 (resume from v1 step 19500)

**Config:** `shoes_224_scratch.yaml` (v2)
**Resume from:** v1 checkpoint `epoch=31-step=19500.ckpt`

Only change: same filter threshold fix as finetune v3. LR unchanged (1e-4 was working well).

---

## Run 3 Results (2026-03-08)

### Run 3a (finetune v3) — FAILED AGAIN

Resumed from v2 step 6500 with LR lowered from 2e-4 → 5e-5. **Same permanent collapse:**
- Steps 6500–~12600: loss oscillating 0.03–0.19, some spikes to 15.5 but recovering
- **Step ~12700: permanent collapse** — loss locked at ~15.5, every step
- Killed at step ~12780

LR 5e-5 only delayed the collapse (step 12.7k vs 6.9k in v2) but did not prevent it.
The DL3DV finetune approach with 6 context views, batch_size=1, and strong anti-fog
losses is fundamentally unstable. **Decision: abandon DL3DV finetune direction.**

### Retrospective: Run 1 Was Actually Stable

Analysis of Run 1 logs revealed the original training was **perfectly stable** (50k steps,
zero spikes > 0.1, smooth loss curve 0.021 → 0.013). The problem was inference, not training.

**Critical CHRONICLE correction:** Run 1 config was incorrectly documented as using
`relative_pose: false` and `pose_norm_method: "none"`. The actual hydra config shows
it used `relative_pose: true`, `pose_norm_method: max_pairwise_d`, and `augment: true`
(inherited from the shoes dataset base config defaults).

Actual Run 1 parameters vs v2/v3:
| Parameter | Run 1 (actual) | v2 | v3 |
|-----------|---------------|----|----|
| LR | **1e-5** | 2e-4 | 5e-5 |
| batch_size | **4** | 1 | 1 |
| context views | **2** | 6 | 6 |
| silhouette | **0.1** | 1.0 | 1.0 |
| opacity | **0.05** | 0.2 | 0.2 |
| augment | **true** | false | false |

Run 1 was stable because: (1) very low LR, (2) batch_size 4 smooths gradients,
(3) weaker loss weights = simpler loss landscape. The v2/v3 explosions were caused
by too many changes at once (LR 5–20× higher + batch 4→1 + loss weights 10× stronger).

### Run 3b (scratch v2) — RUNNING, HEALTHY

Resumed from v1 step 19500. Passed step 25k+ with no issues:
- Loss avg at 25k: ~0.21 (continuing to decrease)
- Speed: 1.01 it/s on L40S
- No spikes, no filter kills
- ETA: ~34h remaining (~Monday 07:00 CET)

**Decision:** Let scratch v2 run to completion (150k steps). This is our best candidate.
DL3DV finetune is abandoned — scratch training with Pi3 backbone is more stable
and avoids the domain mismatch issues entirely.

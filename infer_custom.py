"""
infer_custom.py — run YoNoSplat on a folder of custom images (no poses needed).
Bypasses Hydra config loading entirely — direct encoder/decoder instantiation.

Usage:
    cd /home/lukasz/YoNoSplat
    conda activate yonosplat
    python infer_custom.py \
        --images /path/to/images \
        --checkpoint pretrained_weights/re10k_224x224_ctx2to32.ckpt \
        --output /tmp/yonosplat_out
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf, DictConfig


def load_images(image_dir: Path, size: int = 224) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    )
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    print(f"Found {len(paths)} images: {[p.name for p in paths]}")
    imgs = [transform(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs)  # [v, 3, h, w]


def make_batch(images: torch.Tensor, device: torch.device):
    v, c, h, w = images.shape
    b = 1
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(b, v, 4, 4).clone().to(device)
    f = 0.866
    K = torch.tensor([[f, 0, 0.5], [0, f, 0.5], [0, 0, 1]], dtype=torch.float32)
    intrinsics = K.unsqueeze(0).unsqueeze(0).expand(b, v, 3, 3).clone().to(device)
    images_t = images.unsqueeze(0).to(device)
    near = torch.full((b, v), 0.1, dtype=torch.float32, device=device)
    far = torch.full((b, v), 100.0, dtype=torch.float32, device=device)
    idx = torch.arange(v, device=device).unsqueeze(0)
    views = {"extrinsics": extrinsics, "intrinsics": intrinsics, "image": images_t,
             "near": near, "far": far, "index": idx}
    return {"context": views, "target": views, "scene": ["custom_scene"]}


def build_encoder_cfg() -> DictConfig:
    """Build encoder config directly without Hydra — mirrors yono_re10k experiment."""
    cfg = OmegaConf.create({
        "name": "yonosplat",
        "pose_free": True,
        "pretrained_weights": "",
        "opacity_mapping": {"initial": 0.0, "final": 0.0, "warm_up": 1},
        "num_surfaces": 1,
        "gaussian_adapter": {
            "gaussian_scale_min": 0.5,
            "gaussian_scale_max": 15.0,
            "sh_degree": 0,
        },
        "visualizer": {"num_samples": 8, "min_resolution": 256, "export_ply": False},
        "gt_pose_sampling_decay_start_step": 80000,
        "gt_pose_sampling_decay_end_step": 100000,
        "gt_pose_final_sample_ratio": 0.9,
        "use_checkpoint": True,
        "freeze": "none",
        "input_mean": [0.0, 0.0, 0.0],
        "input_std": [1.0, 1.0, 1.0],
        "gaussian_downsample_ratio": 1,
        "gaussians_per_axis": 14,
        "upscale_token_ratio": 2,
        "backbone": {
            "name": "local_global",
            "intrinsics_embed_degree": 4,
            "intrinsics_embed_type": "pixelwise",
            "predict_intrinsics": True,
            "use_pred_intrinsics_for_embed": False,
        },
    })
    return cfg


def build_decoder_cfg() -> DictConfig:
    return OmegaConf.create({
        "name": "splatting_cuda",
        "background_color": [0.0, 0.0, 0.0],
        "make_scale_invariant": True,
        "prune_opacity_threshold": 0.005,
        "training_prune_ratio": 0.0,
        "training_prune_keep_ratio": 0.1,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default="/tmp/yonosplat_out", type=Path)
    parser.add_argument("--size", default=224, type=int)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Must run from YoNoSplat root so src/ is importable
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, str(Path(__file__).parent))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load images ──────────────────────────────────────────────────────────
    images = load_images(args.images, size=args.size)
    print(f"Images tensor: {images.shape}")

    # ── 2. Build model (no Hydra, no dataset imports) ───────────────────────────
    # Import only encoder/decoder — avoids pulling in dataset/evaluation/skvideo/matplotlib
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder

    enc_cfg = build_encoder_cfg()
    dec_cfg = build_decoder_cfg()

    encoder, _ = get_encoder(enc_cfg)
    decoder = get_decoder(dec_cfg)

    # ── 3. Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    enc_sd = {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    dec_sd = {k[8:]: v for k, v in state_dict.items() if k.startswith("decoder.")}

    if enc_sd:
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        print(f"Encoder — missing: {len(missing)}, unexpected: {len(unexpected)}")
    if dec_sd:
        missing2, unexpected2 = decoder.load_state_dict(dec_sd, strict=False)
        print(f"Decoder — missing: {len(missing2)}, unexpected: {len(unexpected2)}")

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    # ── 4. Build batch & run encoder ───────────────────────────────────────────
    batch = make_batch(images, device)
    print("Running encoder...")
    viz = {}
    with torch.no_grad():
        gaussians = encoder(batch["context"], global_step=0, visualization_dump=viz)

    print(f"Gaussians: means={gaussians.means.shape}, opacities={gaussians.opacities.shape}")

    # ── 5. Export .ply ──────────────────────────────────────────────────────────
    from src.model.ply_export import export_ply
    ply_path = args.output / "gaussians.ply"
    export_mask = gaussians.opacities.squeeze(0) > 0.01
    if not export_mask.any():
        topk = min(4096, gaussians.opacities.shape[1])
        export_indices = gaussians.opacities.squeeze(0).topk(topk).indices
        export_mask = torch.zeros_like(gaussians.opacities.squeeze(0), dtype=torch.bool)
        export_mask[export_indices] = True
    export_ply(
        gaussians.means.squeeze(0)[export_mask],
        gaussians.scales.squeeze(0)[export_mask],
        gaussians.rotations.squeeze(0)[export_mask],
        gaussians.harmonics.squeeze(0)[export_mask],
        gaussians.opacities.squeeze(0)[export_mask],
        path=ply_path,
        shift_and_scale=True,
        save_sh_dc_only=True,
        covariances=gaussians.covariances.squeeze(0)[export_mask],
    )
    print(f"✅ Saved: {ply_path}")

    # ── 6. Render smoke test ────────────────────────────────────────────────────
    print("Running decoder (render back to context views)...")
    v = images.shape[0]
    with torch.no_grad():
        output = decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (args.size, args.size),
        )

    import torchvision
    for i in range(v):
        img = output.color[0, i].clamp(0, 1)
        torchvision.utils.save_image(img, args.output / f"render_{i:02d}.png")
    print(f"✅ Rendered {v} views → {args.output}/render_*.png")
    print("DONE!")


if __name__ == "__main__":
    main()

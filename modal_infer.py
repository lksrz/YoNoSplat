"""
YoNoSplat Inference on Modal.com
---------------------------------
Load a checkpoint from the yonosplat-checkpoints volume, run inference on a shoe
from the dataset volume, and save 4 novel-view renders to /checkpoints/infer_outputs/.

Usage:
  MODAL_PROFILE=wearfits modal run modal_infer.py \
      --ckpt-path /checkpoints/outputs/last.ckpt \
      --shoe-name SHOE_ID

Camera convention: c2w OpenCV (+X right, +Y down, +Z forward), normalised intrinsics.
"""

import modal
import os

# Reuse the same image definition as modal_train.py
yonosplat_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel",
        add_python="3.10",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.0"})
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "TORCH_CUDA_ARCH_LIST": "8.0",
    })
    .apt_install("git", "ninja-build", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install 'numpy<2.0' wheel tqdm hydra-core jaxtyping beartype "
        "wandb einops colorama scikit-image colorspacious matplotlib 'moviepy==1.0.3' "
        "imageio timm dacite lpips plyfile tabulate 'svg.py' scikit-video opencv-python",
        "python -m pip install torchmetrics==1.2.1 pytorch-lightning==2.1.2 lightning==2.1.2 lightning-utilities==0.10.0",
        "python -m pip install e3nn==0.5.1",
    )
    .run_commands(
        "git clone https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git /tmp/rasterizer",
        "cd /tmp/rasterizer && git submodule update --init --recursive",
        "pip install /tmp/rasterizer --no-build-isolation",
        "pip install gsplat",
        gpu="A100",
    )
    .run_commands(
        "git clone https://github.com/lksrz/YoNoSplat.git /opt/YoNoSplat",
        "mkdir -p /opt/YoNoSplat/pretrained_weights",
        "wget -q -O /opt/YoNoSplat/pretrained_weights/pi3.safetensors "
        "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors",
        "wget -q -O /opt/YoNoSplat/pretrained_weights/re10k.ckpt "
        "https://huggingface.co/botaoye/YoNoSplat/resolve/main/re10k_224x224_ctx2to32.ckpt",
        "wget -q -O /opt/YoNoSplat/pretrained_weights/dl3dv.ckpt "
        "https://huggingface.co/botaoye/YoNoSplat/resolve/main/dl3dv_224x224_ctx2to32.ckpt",
    )
)

app = modal.App("yonosplat-infer", image=yonosplat_image)

dataset_vol = modal.Volume.from_name("yonosplat-dataset", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("yonosplat-checkpoints", create_if_missing=True)


@app.function(
    gpu="A100-80GB",
    volumes={
        "/data": dataset_vol,
        "/checkpoints": checkpoints_vol,
    },
    timeout=1800,
)
def infer(
    ckpt_path: str,
    shoe_name: str,
    num_context_views: int = 2,
    num_novel_views: int = 4,
    data_root: str = "/data",
    export_ply_flag: bool = True,
):
    """
    Run inference: load checkpoint, encode context views of `shoe_name`,
    render `num_novel_views` novel views at evenly-spaced azimuth angles,
    and save PNGs to /checkpoints/infer_outputs/<shoe_name>/.
    """
    import json
    import math
    import subprocess
    import sys

    import numpy as np
    import torch
    from PIL import Image
    import torchvision.transforms as tf

    # Pull latest model code
    subprocess.run(["git", "pull", "origin", "main"], cwd="/opt/YoNoSplat", check=True)
    sys.path.insert(0, "/opt/YoNoSplat")

    os.environ["PYTHONPATH"] = "/opt/YoNoSplat"
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    # Hydra must be initialised before importing model code that reads global config.
    # We do a minimal config-load inline rather than invoking main.py.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir="/opt/YoNoSplat/config", job_name="infer"):
        cfg = compose(config_name="main", overrides=[
            "+experiment=shoes_224_finetune",
            f"dataset.shoes.roots=[{data_root}]",
            f"dataset.shoes.view_sampler.num_context_views={num_context_views}",
            "dataset.shoes.view_sampler.num_target_views=1",
        ])

    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.dataset.norm_scale import compute_pose_norm_scale
    from src.misc.cam_utils import camera_normalization

    set_cfg(OmegaConf.to_container(cfg, resolve=True))
    typed_cfg = load_typed_root_config(cfg)
    shoes_cfg = typed_cfg.dataset[0].shoes

    # ------------------------------------------------------------------
    # Load model from checkpoint
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, encoder_visualizer = get_encoder(typed_cfg.model.encoder)
    model = ModelWrapper(
        typed_cfg.optimizer,
        typed_cfg.test,
        typed_cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(typed_cfg.model.decoder),
        [],
        step_tracker=None,
        eval_data_cfg=None,
        gaussian_downsample_ratio=typed_cfg.model.encoder.gaussian_downsample_ratio,
        gaussians_per_axis=typed_cfg.model.encoder.gaussians_per_axis,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print(
        f"Model loaded. missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
    )

    # ------------------------------------------------------------------
    # Load shoe data
    # ------------------------------------------------------------------
    shoe_dir_path = os.path.join(data_root, shoe_name)
    if not os.path.isdir(shoe_dir_path):
        raise FileNotFoundError(f"Shoe directory not found: {shoe_dir_path}")

    poses_path = os.path.join(shoe_dir_path, "poses.json")
    with open(poses_path, "r") as f:
        poses_data = json.load(f)

    print(f"Shoe '{shoe_name}': {len(poses_data)} views available.")

    to_tensor = tf.ToTensor()
    image_shape = tuple(shoes_cfg.input_image_shape)
    source_shape = tuple(shoes_cfg.original_image_shape)

    def load_view(entry):
        """Load view → (img_tensor [3,H,W], c2w [4,4], intr [3,3]) in OpenCV convention."""
        img_path = os.path.join(shoe_dir_path, entry["file_path"])
        with Image.open(img_path) as image:
            img_tensor = to_tensor(image.convert("RGB"))

        # c2w from Blender convention → OpenCV: flip Y and Z
        c2w = torch.tensor(entry["transform_matrix"], dtype=torch.float32)
        if c2w.shape == (3, 4):
            bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
            c2w = torch.cat([c2w, bottom], dim=0)
        c2w = c2w.clone()
        c2w[:3, 1] *= -1  # flip Y
        c2w[:3, 2] *= -1  # flip Z

        k = entry["intrinsics"]
        fx = float(k["fx"])
        fy = float(k["fy"])
        cx = float(k["cx"])
        cy = float(k["cy"])
        if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 2.5:
            fx_norm, fy_norm, cx_norm, cy_norm = fx, fy, cx, cy
        else:
            fx_norm = fx / source_shape[1]
            fy_norm = fy / source_shape[0]
            cx_norm = cx / source_shape[1]
            cy_norm = cy / source_shape[0]
        intr = torch.tensor([
            [fx_norm, 0, cx_norm],
            [0, fy_norm, cy_norm],
            [0, 0, 1],
        ], dtype=torch.float32)

        return img_tensor, c2w, intr

    # Pick context views (first num_context_views)
    ctx_indices = list(range(min(num_context_views, len(poses_data))))
    ctx_imgs, ctx_ext, ctx_intr = [], [], []
    for i in ctx_indices:
        img, ext, intr = load_view(poses_data[i])
        ctx_imgs.append(img)
        ctx_ext.append(ext)
        ctx_intr.append(intr)

    context_extrinsics = torch.stack(ctx_ext)  # (V, 4, 4)
    context_intrinsics = torch.stack(ctx_intr)

    scale = compute_pose_norm_scale(context_extrinsics, shoes_cfg.pose_norm_method)
    scale = float(scale if isinstance(scale, (int, float)) else scale.item())
    scale = max(scale, 1e-6)
    context_extrinsics[:, :3, 3] /= scale
    if shoes_cfg.relative_pose:
        context_extrinsics = camera_normalization(context_extrinsics[0:1], context_extrinsics)

    context_views = apply_crop_shim_to_views(
        {
            "image": torch.stack(ctx_imgs),
            "intrinsics": context_intrinsics,
        },
        image_shape,
    )

    # Build context batch [1, V, ...]
    context_batch = {
        "image": context_views["image"].unsqueeze(0).to(device),        # (1, V, 3, H, W)
        "extrinsics": context_extrinsics.unsqueeze(0).to(device),       # (1, V, 4, 4)
        "intrinsics": context_views["intrinsics"].unsqueeze(0).to(device),  # (1, V, 3, 3)
        "near": (torch.tensor([[0.1] * len(ctx_indices)]) / scale).to(device),    # (1, V)
        "far": (torch.tensor([[100.0] * len(ctx_indices)]) / scale).to(device),   # (1, V)
        "index": torch.tensor([ctx_indices]).to(device),                 # (1, V)
    }

    # ------------------------------------------------------------------
    # Encode context → Gaussians
    # ------------------------------------------------------------------
    print("Encoding context views → Gaussians …")
    with torch.no_grad():
        gaussians = model.encoder(context_batch, global_step=0)

    # ------------------------------------------------------------------
    # Build novel-view cameras at evenly-spaced azimuth angles
    # ------------------------------------------------------------------
    # We compute a rough scene centre from context camera positions and orbit around it.
    ctx_positions = context_extrinsics[:, :3, 3].cpu()  # already normalised
    scene_centre = ctx_positions.mean(dim=0)            # (3,)
    radius = (ctx_positions - scene_centre).norm(dim=-1).mean().item()
    radius = max(radius, 0.5)  # ensure non-zero orbit radius

    def make_orbit_c2w(azimuth_deg: float, elevation_deg: float = 20.0) -> torch.Tensor:
        """
        Create a c2w matrix for a camera on a sphere around scene_centre,
        in OpenCV convention (+X right, +Y down, +Z forward / into scene).
        """
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)

        # Camera position on sphere
        cam_pos = torch.tensor([
            radius * math.cos(el) * math.sin(az),
            -radius * math.sin(el),          # -Y = up in OpenCV
            radius * math.cos(el) * math.cos(az),
        ]) + scene_centre

        # Forward = direction from camera toward scene centre
        fwd = (scene_centre - cam_pos)
        fwd = fwd / (fwd.norm() + 1e-8)

        # Right = fwd × world_up  (world_up = -Y in OpenCV = (0,-1,0))
        world_up = torch.tensor([0.0, -1.0, 0.0])
        right = torch.cross(fwd, world_up)
        right_norm = right.norm()
        if right_norm < 1e-6:
            world_up = torch.tensor([0.0, 0.0, 1.0])
            right = torch.cross(fwd, world_up)
            right_norm = right.norm()
        right = right / right_norm

        # Down = right × fwd  (in OpenCV +Y is down)
        down = torch.cross(right, fwd)

        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = fwd
        c2w[:3, 3] = cam_pos
        return c2w

    # Use intrinsics from the first context view for novel views
    ref_intr = context_views["intrinsics"][0]  # (3, 3)

    azimuths = [i * (360.0 / num_novel_views) for i in range(num_novel_views)]
    novel_extrinsics = torch.stack([make_orbit_c2w(az) for az in azimuths])  # (N, 4, 4)

    novel_batch_ext = novel_extrinsics.unsqueeze(0).to(device)          # (1, N, 4, 4)
    novel_batch_intr = ref_intr.unsqueeze(0).unsqueeze(0).expand(       # (1, N, 3, 3)
        1, num_novel_views, -1, -1
    ).to(device)
    novel_near = (torch.tensor([[0.1] * num_novel_views]) / scale).to(device)
    novel_far = (torch.tensor([[100.0] * num_novel_views]) / scale).to(device)

    # ------------------------------------------------------------------
    # Render novel views
    # ------------------------------------------------------------------
    print(f"Rendering {num_novel_views} novel views …")
    with torch.no_grad():
        output = model.decoder.forward(
            gaussians,
            novel_batch_ext,
            novel_batch_intr,
            novel_near,
            novel_far,
            image_shape,
        )

    # output.color: (1, N, 3, H, W)
    rendered = output.color[0].clamp(0, 1).cpu()  # (N, 3, H, W)

    # ------------------------------------------------------------------
    # Save renders
    # ------------------------------------------------------------------
    out_dir = f"/checkpoints/infer_outputs/{shoe_name}"
    os.makedirs(out_dir, exist_ok=True)

    saved_paths = []
    for i, (img_tensor, az) in enumerate(zip(rendered, azimuths)):
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        fname = os.path.join(out_dir, f"novel_az{int(az):03d}.png")
        pil_img.save(fname)
        saved_paths.append(fname)
        print(f"  Saved: {fname}")

    # ------------------------------------------------------------------
    # Export PLY if requested
    # ------------------------------------------------------------------
    if export_ply_flag:
        from src.model.ply_export import export_ply as save_ply
        from pathlib import Path
        
        ply_path = os.path.join(out_dir, "shoe.ply")
        print(f"Exporting .ply to {ply_path} ...")
        
        # gaussians is a dataclass with fields (batch, gaussian, ...)
        # we take the first item in the batch (b=0)
        save_ply(
            gaussians.means[0],
            gaussians.scales[0],
            gaussians.rotations[0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            Path(ply_path),
            shift_and_scale=True
        )
        saved_paths.append(ply_path)
        print(f"  Saved: {ply_path}")

    # Persist to Modal volume
    checkpoints_vol.commit()

    print(f"\nDone! {len(saved_paths)} renders saved to {out_dir}")
    return saved_paths


@app.local_entrypoint()
def main(
    ckpt_path: str = "/checkpoints/outputs/last.ckpt",
    shoe_name: str = "",
    num_context_views: int = 10,
    num_novel_views: int = 4,
    data_root: str = "/data",
    export_ply: bool = True,
):
    if not shoe_name:
        raise ValueError("--shoe-name is required. Provide a shoe directory name from the dataset volume.")

    print(f"YoNoSplat inference")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  shoe       : {shoe_name}")
    print(f"  context    : {num_context_views} views")
    print(f"  novel      : {num_novel_views} views")
    print(f"  data_root  : {data_root}")
    print(f"  export_ply : {export_ply}")
    print()

    saved = infer.remote(
        ckpt_path=ckpt_path,
        shoe_name=shoe_name,
        num_context_views=num_context_views,
        num_novel_views=num_novel_views,
        data_root=data_root,
        export_ply_flag=export_ply,
    )
    print(f"Saved renders: {saved}")

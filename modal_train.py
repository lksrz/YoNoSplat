"""
YoNoSplat Training on Modal.com
- A100 80GB GPU
- Dataset + checkpoints on Modal Volumes
- Rasterizer compiled at build time

Usage:
  1. Upload dataset:  modal volume put yonosplat-dataset /local/path/to/shoe_renders_final /data
  2. Run training:    modal run modal_train.py
  3. Download ckpts:  modal volume get yonosplat-checkpoints /checkpoints ./local_checkpoints
"""
import modal
import os

# --- Image with all dependencies baked in ---
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
        # Core deps (torch already in base image)
        "python -m pip install 'numpy<2.0' wheel tqdm hydra-core jaxtyping beartype "
        "wandb einops colorama scikit-image colorspacious matplotlib 'moviepy==1.0.3' "
        "imageio timm dacite lpips plyfile tabulate 'svg.py' scikit-video opencv-python",
        "python -m pip install torchmetrics==1.2.1 pytorch-lightning==2.1.2 lightning==2.1.2 lightning-utilities==0.10.0",
        "python -m pip install e3nn==0.5.1",
    )
    # Clone and install the custom rasterizer
    .run_commands(
        "git clone https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git /tmp/rasterizer",
        "cd /tmp/rasterizer && git submodule update --init --recursive",
        "pip install /tmp/rasterizer --no-build-isolation",
        "pip install gsplat",
        gpu="A100",  # Need GPU to compile CUDA kernels
    )
    # Clone YoNoSplat repo
    .run_commands(
        "git clone https://github.com/lksrz/YoNoSplat.git /opt/YoNoSplat",
    )
)

app = modal.App("yonosplat-train", image=yonosplat_image)

# Persistent volumes
dataset_vol = modal.Volume.from_name("yonosplat-dataset", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("yonosplat-checkpoints", create_if_missing=True)
weights_vol = modal.Volume.from_name("yonosplat-weights", create_if_missing=True)


@app.function(
    gpu="A100-80GB",
    volumes={
        "/data": dataset_vol,
        "/checkpoints": checkpoints_vol,
        "/pretrained_weights": weights_vol,
    },
    timeout=86400,  # 24h max

)
def train(
    max_steps: int = 50000,
    batch_size: int = 4,
    num_workers: int = 4,
    num_context_views: int = 2,
    num_target_views: int = 1,
    resume_from: str = None,
    wandb_key: str = None,
):
    import subprocess
    import torch

    print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Pull latest code
    subprocess.run(
        ["git", "pull", "origin", "main"],
        cwd="/opt/YoNoSplat",
        check=True,
    )

    # Check dataset
    data_root = "/data"
    shoe_dirs = [d for d in os.listdir(data_root)
                 if os.path.isdir(os.path.join(data_root, d))
                 and os.path.exists(os.path.join(data_root, d, "poses.json"))]
    print(f"Dataset: {len(shoe_dirs)} complete shoes in {data_root}")
    if len(shoe_dirs) == 0:
        raise RuntimeError(
            "No shoes found! Upload dataset first:\n"
            "  modal volume put yonosplat-dataset /path/to/shoe_renders_final/ /"
        )

    # Setup wandb (optional)
    wandb_mode = "disabled"
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        wandb_mode = "online"

    # Build hydra overrides
    overrides = [
        f"dataset.shoes.roots=[{data_root}]",
        f"dataset.shoes.view_sampler.num_context_views={num_context_views}",
        f"dataset.shoes.view_sampler.num_target_views={num_target_views}",
        f"data_loader.train.batch_size={batch_size}",
        f"data_loader.train.num_workers={num_workers}",
        f"trainer.max_steps={max_steps}",
        f"wandb.mode={wandb_mode}",
        # Save checkpoints to persistent volume
        "checkpointing.save_weights_only=true",
        "checkpointing.every_n_train_steps=500",
    ]

    if resume_from:
        overrides.append(f"checkpointing.load={resume_from}")

    cmd = [
        "python", "-u", "src/main.py",
        "--config-name", "main_smoke",
    ] + overrides

    print(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/YoNoSplat"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    # Symlink checkpoint output dir to persistent volume
    ckpt_link = "/opt/YoNoSplat/outputs"
    os.makedirs("/checkpoints/outputs", exist_ok=True)
    if os.path.exists(ckpt_link) and not os.path.islink(ckpt_link):
        subprocess.run(["rm", "-rf", ckpt_link])
    if not os.path.exists(ckpt_link):
        os.symlink("/checkpoints/outputs", ckpt_link)

    result = subprocess.run(
        cmd,
        cwd="/opt/YoNoSplat",
        env=env,
    )

    # Commit volumes to persist checkpoints
    checkpoints_vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print("Training complete!")


@app.function(
    volumes={"/data": dataset_vol},
    timeout=300,
)
def check_dataset():
    """Quick check: how many shoes are uploaded and ready."""
    data_root = "/data"
    total = 0
    complete = 0
    for d in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, d)
        if os.path.isdir(full):
            total += 1
            if os.path.exists(os.path.join(full, "poses.json")):
                complete += 1

    print(f"Total dirs: {total}, Complete (with poses.json): {complete}")
    return {"total": total, "complete": complete}


@app.local_entrypoint()
def main(
    max_steps: int = 50000,
    batch_size: int = 4,
    num_workers: int = 4,
    resume_from: str = None,
    wandb_key: str = None,
    check_only: bool = False,
):
    if check_only:
        result = check_dataset.remote()
        print(f"Dataset status: {result}")
        return

    print(f"Starting YoNoSplat training on Modal A100-80GB")
    print(f"  max_steps={max_steps}, batch_size={batch_size}")
    print(f"  Resume: {resume_from or 'from scratch'}")

    train.remote(
        max_steps=max_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        resume_from=resume_from,
        wandb_key=wandb_key,
    )

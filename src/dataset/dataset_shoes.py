import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset
from .norm_scale import compute_pose_norm_scale
import numpy as np

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler

logger = logging.getLogger(__name__)

@dataclass
class DatasetShoesCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    relative_pose: bool
    augment: bool
    skip_bad_shape: bool
    pose_norm_method: str = "max_pairwise_d"
    val_fraction: float = 0.1

@dataclass
class DatasetShoesCfgWrapper:
    shoes: DatasetShoesCfg

class DatasetShoes(Dataset):
    cfg: DatasetShoesCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: DatasetShoesCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.name = cfg.name

        # Find all shoe directories that have a completed poses.json
        self.shoe_dirs = []
        skipped = 0
        for root in cfg.roots:
            if not root.exists():
                logger.warning(f"Root {root} does not exist")
                continue
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "poses.json").exists():
                    self.shoe_dirs.append(d)
                elif d.is_dir():
                    skipped += 1
        
        if skipped:
            logger.info(f"Skipped {skipped} incomplete shoes (no poses.json)")
        logger.info(f"Found {len(self.shoe_dirs)} complete shoes in {cfg.roots}")

        # Deterministic train/val split — sort globally, last val_fraction% = val
        self.shoe_dirs = sorted(self.shoe_dirs)
        n_total = len(self.shoe_dirs)
        n_val = max(1, int(round(n_total * cfg.val_fraction)))
        n_train = n_total - n_val
        if stage == "val":
            self.shoe_dirs = self.shoe_dirs[n_train:]
        elif stage == "train":
            self.shoe_dirs = self.shoe_dirs[:n_train]
        # "test" keeps all dirs unchanged
        logger.info(
            f"Stage={stage}: using {len(self.shoe_dirs)}/{n_total} shoes "
            f"(val_fraction={cfg.val_fraction})"
        )

    def __len__(self):
        return len(self.shoe_dirs)

    def _load_view(self, entry, shoe_dir):
        """Load a single view: image tensor, w2c extrinsic, intrinsic matrix."""
        img_path = shoe_dir / entry['file_path']
        image = Image.open(img_path).convert("RGB")

        # Resize if needed (e.g. 512 -> 518 for DINOv2 patch_size=14)
        if self.cfg.input_image_shape:
            image = image.resize((self.cfg.input_image_shape[1], self.cfg.input_image_shape[0]))

        img_tensor = self.to_tensor(image)

        # Extrinsics: YoNoSplat/pixelSplat convention = camera-to-world (c2w)
        # OpenCV-style: +X right, +Y down, +Z forward
        c2w = torch.tensor(entry['transform_matrix'], dtype=torch.float32)
        # Ensure 4x4 matrix
        if c2w.shape == (3, 4):
            bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
            c2w = torch.cat([c2w, bottom], dim=0)
        # Blender convention: +X right, +Y up, +Z backward
        # Convert to OpenCV: flip Y and Z axes
        c2w[:3, 1] *= -1  # flip Y
        c2w[:3, 2] *= -1  # flip Z

        # Intrinsics: NORMALIZED (fx/width, fy/height, cx/width, cy/height)
        # pixelSplat convention: row 0 divided by image width, row 1 by height
        k = entry['intrinsics']
        img_w, img_h = image.size  # after resize
        intr = torch.tensor([
            [k['fx'] / img_w, 0, k['cx'] / img_w],
            [0, k['fy'] / img_h, k['cy'] / img_h],
            [0, 0, 1]
        ], dtype=torch.float32)

        return img_tensor, c2w, intr

    def __getitem__(self, index):
        # MixedBatchSampler yields (idx, num_context_views) tuples
        if isinstance(index, (tuple, list)):
            index, num_context = index
        else:
            num_context = self.view_sampler.cfg.num_context_views

        shoe_dir = self.shoe_dirs[index]
        poses_path = shoe_dir / "poses.json"

        with open(poses_path, 'r') as f:
            poses_data = json.load(f)

        num_available = len(poses_data)
        num_target = self.view_sampler.cfg.num_target_views

        # Randomly sample context + target indices (no overlap)
        all_indices = np.arange(num_available)
        np.random.shuffle(all_indices)
        context_indices = sorted(all_indices[:num_context].tolist())
        target_indices = sorted(all_indices[num_context:num_context + num_target].tolist())

        # Load only the selected views
        ctx_imgs, ctx_ext, ctx_intr = [], [], []
        for i in context_indices:
            img, ext, intr = self._load_view(poses_data[i], shoe_dir)
            ctx_imgs.append(img)
            ctx_ext.append(ext)
            ctx_intr.append(intr)

        tgt_imgs, tgt_ext, tgt_intr = [], [], []
        for i in target_indices:
            img, ext, intr = self._load_view(poses_data[i], shoe_dir)
            tgt_imgs.append(img)
            tgt_ext.append(ext)
            tgt_intr.append(intr)

        # Pose normalization (max pairwise distance) — critical per paper
        context_extrinsics = torch.stack(ctx_ext)
        target_extrinsics = torch.stack(tgt_ext)
        all_extrinsics = torch.cat([context_extrinsics, target_extrinsics], dim=0)
        
        scale = compute_pose_norm_scale(context_extrinsics, "max_pairwise_d")
        if isinstance(scale, (int, float)):
            scale = max(scale, 1e-6)  # avoid division by zero
        else:
            scale = torch.clamp(scale, min=1e-6)
        
        context_extrinsics[:, :3, 3] /= scale
        target_extrinsics[:, :3, 3] /= scale

        return {
            "context": {
                "image": torch.stack(ctx_imgs),
                "extrinsics": context_extrinsics,
                "intrinsics": torch.stack(ctx_intr),
                "index": torch.tensor(context_indices, dtype=torch.long),
                "near": torch.tensor([0.1] * num_context),
                "far": torch.tensor([100.0] * num_context),
            },
            "target": {
                "image": torch.stack(tgt_imgs),
                "extrinsics": target_extrinsics,
                "intrinsics": torch.stack(tgt_intr),
                "index": torch.tensor(target_indices, dtype=torch.long),
                "near": torch.tensor([0.1] * num_target),
                "far": torch.tensor([100.0] * num_target),
            },
            "scene": shoe_dir.name
        }

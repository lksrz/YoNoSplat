import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset
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

        # Find all shoe directories
        self.shoe_dirs = []
        for root in cfg.roots:
            if not root.exists():
                logger.warning(f"Root {root} does not exist")
                continue
            subdirs = [d for d in root.iterdir() if d.is_dir()]
            self.shoe_dirs.extend(subdirs)
        
        logger.info(f"Found {len(self.shoe_dirs)} shoes in {cfg.roots}")

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

        # Transform matrix is camera-to-world → invert to world-to-camera
        c2w = torch.tensor(entry['transform_matrix'], dtype=torch.float32)
        w2c = torch.inverse(c2w)

        k = entry['intrinsics']
        intr = torch.tensor([
            [k['fx'], 0, k['cx']],
            [0, k['fy'], k['cy']],
            [0, 0, 1]
        ], dtype=torch.float32)

        return img_tensor, w2c, intr

    def __getitem__(self, index):
        shoe_dir = self.shoe_dirs[index]
        poses_path = shoe_dir / "poses.json"

        with open(poses_path, 'r') as f:
            poses_data = json.load(f)

        num_available = len(poses_data)
        num_context = self.view_sampler.cfg.num_context_views
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

        return {
            "context": {
                "image": torch.stack(ctx_imgs),
                "extrinsics": torch.stack(ctx_ext),
                "intrinsics": torch.stack(ctx_intr),
                "index": torch.tensor(context_indices, dtype=torch.long),
                "near": torch.tensor([0.1] * num_context),
                "far": torch.tensor([100.0] * num_context),
            },
            "target": {
                "image": torch.stack(tgt_imgs),
                "extrinsics": torch.stack(tgt_ext),
                "intrinsics": torch.stack(tgt_intr),
                "index": torch.tensor(target_indices, dtype=torch.long),
                "near": torch.tensor([0.1] * num_target),
                "far": torch.tensor([100.0] * num_target),
            },
            "scene": shoe_dir.name
        }

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

    def __getitem__(self, index):
        shoe_dir = self.shoe_dirs[index]
        poses_path = shoe_dir / "poses.json"
        
        with open(poses_path, 'r') as f:
            poses_data = json.load(f)
            
        # Select views according to view_sampler
        # For simplicity in this local version, we take all or a subset
        num_available = len(poses_data)
        
        # This is a bit simplified compared to the IterableDataset version
        # but should work for the ModelWrapper.
        
        images = []
        extrinsics = []
        intrinsics = []
        
        for entry in poses_data:
            img_path = shoe_dir / entry['file_path']
            image = Image.open(img_path).convert("RGB")
            
            # Resize if needed
            if self.cfg.input_image_shape:
                image = image.resize((self.cfg.input_image_shape[1], self.cfg.input_image_shape[0]))
            
            images.append(self.to_tensor(image))
            
            # Transform matrix is camera-to-world
            c2w = torch.tensor(entry['transform_matrix'], dtype=torch.float32)
            # We need world-to-camera for some parts of YoNoSplat
            w2c = torch.inverse(c2w)
            extrinsics.append(w2c)
            
            # Intrinsics
            k = entry['intrinsics']
            intrinsics.append(torch.tensor([
                [k['fx'], 0, k['cx']],
                [0, k['fy'], k['cy']],
                [0, 0, 1]
            ], dtype=torch.float32))

        return {
            "context": {
                "image": torch.stack(images[:self.view_sampler.cfg.num_context_views]),
                "extrinsics": torch.stack(extrinsics[:self.view_sampler.cfg.num_context_views]),
                "intrinsics": torch.stack(intrinsics[:self.view_sampler.cfg.num_context_views]),
                "index": torch.arange(self.view_sampler.cfg.num_context_views),
            },
            "target": {
                "image": torch.stack(images[self.view_sampler.cfg.num_context_views:]),
                "extrinsics": torch.stack(extrinsics[self.view_sampler.cfg.num_context_views:]),
                "intrinsics": torch.stack(intrinsics[self.view_sampler.cfg.num_context_views:]),
                "index": torch.arange(self.view_sampler.cfg.num_context_views, num_available),
            },
            "scene": shoe_dir.name
        }

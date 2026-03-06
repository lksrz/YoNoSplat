import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from ..geometry.projection import get_fov
from ..misc.cam_utils import camera_normalization
from .dataset import DatasetCfgCommon
from .norm_scale import compute_pose_norm_scale
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

logger = logging.getLogger(__name__)


@dataclass
class DatasetShoesCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float = 1e-3
    baseline_max: float = 1e10
    max_fov: float = 100.0
    make_baseline_1: bool = True
    relative_pose: bool = True
    augment: bool = True
    skip_bad_shape: bool = True
    pose_norm_method: str = "max_pairwise_d"
    val_fraction: float = 0.1


@dataclass
class DatasetShoesCfgWrapper:
    shoes: DatasetShoesCfg


@dataclass
class ShoeView:
    image_path: Path
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor


@dataclass
class ShoeScene:
    name: str
    views: list[ShoeView]


class DatasetShoes(Dataset):
    cfg: DatasetShoesCfg
    stage: Stage
    view_sampler: ViewSampler

    near: float = 0.1
    far: float = 100.0

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

        scenes: list[ShoeScene] = []
        skipped = 0
        for root in cfg.roots:
            if not root.exists():
                logger.warning(f"Root {root} does not exist")
                continue
            for shoe_dir in sorted(root.iterdir()):
                if not shoe_dir.is_dir():
                    continue
                scene = self._load_scene(shoe_dir)
                if scene is None:
                    skipped += 1
                    continue
                scenes.append(scene)

        if skipped:
            logger.info(f"Skipped {skipped} invalid or incomplete shoes")

        scenes = sorted(scenes, key=lambda scene: scene.name)
        total_scenes = len(scenes)
        val_scenes = 0
        if total_scenes > 1 and cfg.val_fraction > 0:
            val_scenes = max(1, int(round(total_scenes * cfg.val_fraction)))
            val_scenes = min(val_scenes, total_scenes - 1)

        if stage == "val":
            self.scenes = scenes[total_scenes - val_scenes :]
        elif stage == "train":
            self.scenes = scenes[: total_scenes - val_scenes]
        else:
            self.scenes = scenes

        logger.info(
            f"Stage={stage}: using {len(self.scenes)}/{total_scenes} shoes "
            f"(val_fraction={cfg.val_fraction})"
        )

    def _load_scene(self, shoe_dir: Path) -> ShoeScene | None:
        poses_path = shoe_dir / "poses.json"
        if not poses_path.exists():
            return None

        try:
            with poses_path.open("r") as f:
                poses_data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Skipped {shoe_dir.name}: could not read poses.json ({exc})")
            return None

        if not isinstance(poses_data, list):
            logger.warning(f"Skipped {shoe_dir.name}: poses.json must be a list of views")
            return None

        views: list[ShoeView] = []
        for entry in poses_data:
            view = self._parse_view(entry, shoe_dir)
            if view is not None:
                views.append(view)

        min_required_views = self._min_required_views()
        if len(views) < min_required_views:
            logger.warning(
                f"Skipped {shoe_dir.name}: only {len(views)} valid views, need at least {min_required_views}"
            )
            return None

        return ShoeScene(name=shoe_dir.name, views=views)

    def _parse_view(self, entry: dict, shoe_dir: Path) -> ShoeView | None:
        try:
            image_path = shoe_dir / entry["file_path"]
            if not image_path.exists():
                return None

            c2w = torch.tensor(entry["transform_matrix"], dtype=torch.float32)
            if c2w.shape == (3, 4):
                bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
                c2w = torch.cat([c2w, bottom], dim=0)
            if c2w.shape != (4, 4):
                return None

            # Blender c2w -> OpenCV c2w.
            c2w = c2w.clone()
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1

            intrinsics = entry["intrinsics"]
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
        except (KeyError, TypeError, ValueError):
            return None

        src_h, src_w = self.cfg.original_image_shape
        if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 2.5:
            fx_norm, fy_norm, cx_norm, cy_norm = fx, fy, cx, cy
        else:
            fx_norm = fx / src_w
            fy_norm = fy / src_h
            cx_norm = cx / src_w
            cy_norm = cy / src_h

        k = torch.tensor(
            [
                [fx_norm, 0.0, cx_norm],
                [0.0, fy_norm, cy_norm],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return ShoeView(image_path=image_path, extrinsics=c2w, intrinsics=k)

    def _min_required_views(self) -> int:
        context_views = self.view_sampler.cfg.num_context_views
        if isinstance(context_views, list):
            context_views = context_views[-1]
        return context_views + self.view_sampler.cfg.num_target_views

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_images(
        self,
        scene: ShoeScene,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load images and alpha masks. Returns (images [v,3,H,W], masks [v,1,H,W])."""
        images = []
        masks = []
        for index in indices.tolist():
            with Image.open(scene.views[index].image_path) as image:
                if image.mode == "RGBA":
                    rgba = self.to_tensor(image)           # [4, H, W]
                    rgb = rgba[:3]                          # [3, H, W]
                    alpha = rgba[3:4]                       # [1, H, W]
                    # Composite on white background for network input
                    rgb = rgb * alpha + (1.0 - alpha)
                    images.append(rgb)
                    masks.append(alpha)
                else:
                    image_tensor = self.to_tensor(image.convert("RGB"))
                    images.append(image_tensor)
                    masks.append(torch.ones(1, image_tensor.shape[1], image_tensor.shape[2]))
        images = torch.stack(images)
        masks = torch.stack(masks)

        expected_shape = (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and images.shape[1:] != expected_shape:
            raise ValueError(
                f"Bad image shape for {scene.name}: expected {expected_shape}, got {tuple(images.shape[1:])}"
            )
        return images, masks

    def _build_example(
        self,
        scene: ShoeScene,
        num_context_views: int | None,
    ) -> dict:
        extrinsics = torch.stack([view.extrinsics for view in scene.views])
        intrinsics = torch.stack([view.intrinsics for view in scene.views])

        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise ValueError(f"Scene {scene.name} has field of view above {self.cfg.max_fov}")

        context_indices, target_indices, overlap = self.view_sampler.sample(
            scene.name,
            extrinsics,
            intrinsics,
            num_context_views=num_context_views,
        )
        context_indices = context_indices.to(dtype=torch.long, device=torch.device("cpu"))
        target_indices = target_indices.to(dtype=torch.long, device=torch.device("cpu"))

        context_images, context_masks = self._load_images(scene, context_indices)
        target_images, target_masks = self._load_images(scene, target_indices)

        context_extrinsics = extrinsics[context_indices].clone()
        target_extrinsics = extrinsics[target_indices].clone()
        num_context_images = len(context_extrinsics)
        all_used_extrinsics = torch.cat([context_extrinsics, target_extrinsics], dim=0)

        scale = compute_pose_norm_scale(context_extrinsics, self.cfg.pose_norm_method)
        scale = float(scale if isinstance(scale, (int, float)) else scale.item())
        scale = max(scale, 1e-6)
        if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
            raise ValueError(
                f"Scene {scene.name} baseline out of range: {scale:.6f}"
            )
        all_used_extrinsics[:, :3, 3] /= scale

        if self.cfg.relative_pose:
            all_used_extrinsics = camera_normalization(
                all_used_extrinsics[0:1],
                all_used_extrinsics,
            )

        example = {
            "context": {
                "extrinsics": all_used_extrinsics[:num_context_images],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "mask": context_masks,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "overlap": overlap.cpu(),
            },
            "target": {
                "extrinsics": all_used_extrinsics[num_context_images:],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "mask": target_masks,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": scene.name,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        return apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    def __getitem__(self, index):
        if len(self.scenes) == 0:
            raise IndexError("No valid shoe scenes were found")

        num_context_views = None
        if isinstance(index, (tuple, list)):
            index, num_context_views = index

        for retry in range(min(8, len(self.scenes))):
            scene = self.scenes[(index + retry) % len(self.scenes)]
            try:
                return self._build_example(scene, num_context_views)
            except (IndexError, OSError, ValueError) as exc:
                logger.warning(f"Skipped scene {scene.name}: {exc}")
                continue

        raise RuntimeError(f"Failed to load a valid sample after retries from index {index}")

    def get_bound(self, bound: str, num_views: int) -> torch.Tensor:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return value.repeat(num_views)

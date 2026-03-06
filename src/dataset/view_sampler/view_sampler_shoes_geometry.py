from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerShoesGeometryCfg:
    name: Literal["shoes_geometry"]
    num_context_views: int | list[int]
    num_target_views: int
    min_context_angle_deg: float = 20.0
    max_context_angle_deg: float = 60.0
    fallback_min_context_angle_deg: float = 10.0
    fallback_max_context_angle_deg: float = 90.0
    min_target_angle_deg: float = 5.0
    max_target_angle_deg: float = 25.0
    max_img_per_gpu: int = 8


class ViewSamplerShoesGeometry(ViewSampler[ViewSamplerShoesGeometryCfg]):
    def _resolve_num_context_views(
        self,
        device: torch.device,
        num_context_views: int | None,
    ) -> int:
        if num_context_views is not None:
            return num_context_views
        if isinstance(self.cfg.num_context_views, list):
            assert len(self.cfg.num_context_views) == 2
            generator = torch.Generator(device=device)
            generator.manual_seed(self.global_step)
            return torch.randint(
                self.cfg.num_context_views[0],
                self.cfg.num_context_views[1] + 1,
                size=(),
                generator=generator,
                device=device,
            ).item()
        return self.cfg.num_context_views

    def _angles_to(
        self,
        directions: Float[Tensor, "view 3"],
        reference: Float[Tensor, "num_ref 3"],
    ) -> Float[Tensor, "view num_ref"]:
        cosine = directions @ reference.T
        cosine = cosine.clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cosine))

    def _candidate_mask(
        self,
        angles: Float[Tensor, "view"],
        lower: float,
        upper: float,
        forbidden: Tensor,
    ) -> Tensor:
        mask = (angles >= lower) & (angles <= upper)
        mask[forbidden] = False
        return mask

    def _choose_index(
        self,
        candidates: Tensor,
        device: torch.device,
    ) -> int:
        if len(candidates) == 0:
            raise ValueError("No candidates available for geometry-aware sampling.")
        if self.stage == "train":
            return candidates[torch.randint(len(candidates), (), device=device)].item()
        return candidates[0].item()

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        num_context_views: int | None = None,
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
        Float[Tensor, " overlap"],
    ]:
        del scene, intrinsics

        num_views = extrinsics.shape[0]
        num_context = self._resolve_num_context_views(device, num_context_views)
        num_target = self.cfg.num_target_views

        if num_views < num_context + num_target:
            raise ValueError("Example does not have enough views.")

        camera_centers = extrinsics[:, :3, 3]
        scene_center = camera_centers.mean(dim=0, keepdim=True)
        directions = camera_centers - scene_center
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.stage == "train":
            anchor = torch.randint(num_views, (), device=device).item()
        else:
            anchor = 0

        selected = [anchor]
        selected_mask = torch.zeros(num_views, dtype=torch.bool, device=device)
        selected_mask[anchor] = True

        anchor_angles = self._angles_to(directions, directions[[anchor]]).squeeze(-1)
        pair_mask = self._candidate_mask(
            anchor_angles,
            self.cfg.min_context_angle_deg,
            self.cfg.max_context_angle_deg,
            selected_mask,
        )
        if not pair_mask.any():
            pair_mask = self._candidate_mask(
                anchor_angles,
                self.cfg.fallback_min_context_angle_deg,
                self.cfg.fallback_max_context_angle_deg,
                selected_mask,
            )
        if pair_mask.any():
            pair_candidates = torch.nonzero(pair_mask, as_tuple=False).flatten()
        else:
            pair_candidates = torch.argsort(anchor_angles, descending=True)
            pair_candidates = pair_candidates[~selected_mask[pair_candidates]]
        second = self._choose_index(pair_candidates, device)
        selected.append(second)
        selected_mask[second] = True

        while len(selected) < num_context:
            remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
            remaining_angles = self._angles_to(directions[remaining], directions[selected])
            best_idx = torch.argmax(remaining_angles.min(dim=1).values).item()
            chosen = remaining[best_idx].item()
            selected.append(chosen)
            selected_mask[chosen] = True

        context_indices = torch.tensor(selected, dtype=torch.int64, device=device)

        remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        if len(remaining) < num_target:
            raise ValueError("Not enough remaining views to sample targets.")

        context_dirs = directions[context_indices]
        mean_direction = context_dirs.mean(dim=0, keepdim=True)
        if mean_direction.norm() < 1e-6:
            mean_direction = context_dirs[:1]
        mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        remaining_dirs = directions[remaining]
        min_context_angle = self._angles_to(remaining_dirs, context_dirs).min(dim=1).values
        mean_angle = self._angles_to(remaining_dirs, mean_direction).squeeze(-1)

        preferred = (min_context_angle >= self.cfg.min_target_angle_deg) & (
            min_context_angle <= self.cfg.max_target_angle_deg
        )
        candidates = remaining[preferred]
        if len(candidates) < num_target:
            candidates = remaining

        candidate_mean_angles = mean_angle[torch.searchsorted(remaining, candidates)]
        order = torch.argsort(candidate_mean_angles)
        candidates = candidates[order]
        if self.stage == "train" and len(candidates) > num_target:
            shuffled = candidates[torch.randperm(len(candidates), device=device)]
            candidates = shuffled[: max(num_target * 2, num_target)]
            candidate_mean_angles = mean_angle[torch.searchsorted(remaining, candidates)]
            candidates = candidates[torch.argsort(candidate_mean_angles)]

        target_indices = candidates[:num_target].to(torch.int64)
        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)
        return context_indices, target_indices, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views

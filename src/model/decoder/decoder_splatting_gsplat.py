from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians
from .decoder import Decoder, DecoderOutput
from .prune import prune_gaussians
from math import sqrt
from gsplat import rasterization

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


@dataclass
class DecoderSplattingSplatCfg:
    name: Literal["splatting_gsplat"]
    background_color: list[float]
    make_scale_invariant: bool
    prune_opacity_threshold: float = 0.005
    training_prune_ratio: float = 0.
    training_prune_keep_ratio: float = 0.1
    bounds_radius: float = -1.0


class DecoderSplattingGSPlat(Decoder[DecoderSplattingSplatCfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
            self,
            cfg: DecoderSplattingSplatCfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def rendering_fn(
            self,
            gaussians: Gaussians,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            depth_mode: DepthRenderingMode | None = None,
            cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
            cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
            bounds_radius: Tensor | float | None = None,
    ) -> DecoderOutput:
        gaussians = prune_gaussians(
            gaussians,
            self.cfg.prune_opacity_threshold,
            self.cfg.training_prune_ratio,
            self.cfg.training_prune_keep_ratio,
            inference=not self.training,
            bounds_radius=bounds_radius if bounds_radius is not None else self.cfg.bounds_radius,
        )

        B, V, _, _ = intrinsics.shape
        H, W = image_shape
        means, opacitys, rotations, scales, features = gaussians.means, gaussians.opacities, gaussians.rotations, gaussians.scales, gaussians.harmonics.permute(
            0, 1, 3, 2).contiguous()
        covars = gaussians.covariances

        w2c = extrinsics.float().inverse()  # (B, V, 4, 4)
        sh_degree = (int(sqrt(features.shape[-2])) - 1)

        intrinsics_denorm = intrinsics.clone()
        intrinsics_denorm[:, :, 0] = intrinsics_denorm[:, :, 0] * W
        intrinsics_denorm[:, :, 1] = intrinsics_denorm[:, :, 1] * H

        backgrounds = self.background_color.unsqueeze(0).unsqueeze(0).repeat(B, V, 1)

        rendering, alpha, _ = rasterization(means, rotations, scales, opacitys, features,
                                            w2c,
                                            intrinsics_denorm,
                                            W, H,
                                            sh_degree=sh_degree,
                                            render_mode="RGB+D", packed=False,
                                            backgrounds=backgrounds,
                                            radius_clip=0.1,
                                            covars=covars,
                                            rasterize_mode='classic',
                                            )  # (V, H, W, 3)
        rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
        rendering_img = rendering_img.clamp(0.0, 1.0)
        if alpha.ndim == 4:
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.permute(0, 1, 4, 2, 3).clamp(0.0, 1.0)
        return DecoderOutput(
            rendering_img.permute(0, 1, 4, 2, 3),
            rendering_depth.squeeze(-1),
            alpha,
        )

    def forward(
            self,
            gaussians: Gaussians,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            depth_mode: DepthRenderingMode | None = None,
            cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
            cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:

        return self.rendering_fn(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode, cam_rot_delta,
                                 cam_trans_delta)

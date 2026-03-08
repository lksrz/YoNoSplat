import random
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from einops.layers.torch import Rearrange

from .backbone.dinov2.layers import PatchEmbed
from .backbone.croco.misc import freeze_all_params
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from ..types import Gaussians
from .backbone import BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from .layers.transformer_head import TransformerDecoder, LinearPts3d, CrossAttentionDecoder
from .layers.camera_head import CameraHead
from ...loss.loss_pose import se3_inverse
from ...misc.schedule_sample import get_scheduled_sampling_epsilon

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderYoNoSplatCfg:
    name: Literal["yonosplat"]
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    num_surfaces: int
    input_mean: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0)
    input_std: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0)
    pretrained_weights: str = ""
    pose_free: bool = True

    use_checkpoint: bool = False
    freeze: str = 'none'

    gt_pose_sampling_decay_start_step: int = 1000
    gt_pose_sampling_decay_end_step: int = 5000
    gt_pose_final_sample_ratio: float = 0.9

    gaussian_downsample_ratio: int = 1
    gaussians_per_axis: int = 14
    upscale_token_ratio: int = 1



def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderYoNoSplat(Encoder[EncoderYoNoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderYoNoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3, use_checkpoint=cfg.use_checkpoint)

        self.pose_free = cfg.pose_free
        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_size
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gaussian_downsample_ratio = cfg.gaussian_downsample_ratio
        self.gaussians_per_axis = cfg.gaussians_per_axis
        self.gaussians_per_axis = min(self.gaussians_per_axis, self.patch_size // self.gaussian_downsample_ratio)

        self.upscale_token_ratio = cfg.upscale_token_ratio
        self.head_pathch_size = self.patch_size // self.upscale_token_ratio
        self.position_getter = self.backbone.position_getter

        self.dec_embed_dim = 1024
        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.backbone.rope,
            use_checkpoint=cfg.use_checkpoint,
        )
        self.point_head = LinearPts3d(patch_size=self.patch_size / self.upscale_token_ratio, dec_embed_dim=1024, output_dim=3, downsample_ratio=self.gaussian_downsample_ratio, points_per_axis=self.gaussians_per_axis // self.upscale_token_ratio)

        # ----------------------
        #     Gaussian Parameters Decoder
        # ----------------------
        self.gaussian_decoder = deepcopy(self.point_decoder)
        self.gaussian_head = LinearPts3d(patch_size=self.patch_size / self.upscale_token_ratio, dec_embed_dim=1024, output_dim=self.raw_gs_dim, downsample_ratio=self.gaussian_downsample_ratio, points_per_axis=self.gaussians_per_axis // self.upscale_token_ratio)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.backbone.rope,
            use_checkpoint=cfg.use_checkpoint,
        )
        self.camera_head = CameraHead(dim=512)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.rgb_embed = PatchEmbed(patch_size=self.patch_size // self.upscale_token_ratio, in_chans=3, embed_dim=2048, norm_layer=norm_layer)
        nn.init.constant_(self.rgb_embed.proj.weight, 0)
        nn.init.constant_(self.rgb_embed.proj.bias, 0)

        # freeze parameters
        self.set_freeze(cfg.freeze)

    def set_freeze(self, freeze):  # this is for use by downstream models
        if freeze == 'none':
            return

        to_be_frozen = {
            'none':     [],
            'encoder':     [self.backbone.encoder],
            'decoder':     [self.backbone.decoder, self.backbone.register_token, self.backbone.intrinsics_embed_layer],
            'encoder+decoder': [self.backbone],
            "heads": [self.point_decoder, self.point_head, self.gaussian_decoder, self.gaussian_head, self.rgb_embed],
            'encoder+decoder+point_head': [self.backbone, self.point_decoder, self.point_head, self.gaussian_decoder, self.gaussian_head, self.rgb_embed],
            'all': [self]
        }
        freeze_all_params(to_be_frozen[freeze])

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        # Encode the context images.
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            hidden, pos, patch_start_idx, x_low, intrinsic_pred = self.backbone(context["image"], context["intrinsics"].clone())
            del x_low

            # hidden shape: (b*v, n, c), pos shape: (b*v, n, 2)
            if self.upscale_token_ratio > 1:
                hidden_aux_token = hidden[:, :patch_start_idx, :]
                hidden_img_token = hidden[:, patch_start_idx:, :]
                hidden_img_token = rearrange(hidden_img_token, "b (h w) c -> b c h w", h=patch_h, w=patch_w)
                hidden_img_token = F.interpolate(hidden_img_token, scale_factor=self.upscale_token_ratio, mode="bilinear", align_corners=False)
                hidden_img_token = rearrange(hidden_img_token, "b c h w -> b (h w) c")
                hidden_upsampled = torch.cat([hidden_aux_token, hidden_img_token], dim=1)

                pos_aux = pos[:, :patch_start_idx]
                pos_img = self.position_getter(b * v, patch_h * self.upscale_token_ratio, patch_w * self.upscale_token_ratio, device=device)
                pos_img = pos_img + 1 if patch_start_idx > 0 else pos_img
                pos_upsampled = torch.cat([pos_aux, pos_img], dim=1)
            else:
                hidden_upsampled = hidden
                pos_upsampled = pos

            rgb = rearrange(context['image'], 'b v c h w -> (b v) c h w')
            rgb_feat = self.rgb_embed(rgb)
            hidden_gaussian = hidden_upsampled.clone()
            hidden_gaussian[:, patch_start_idx:, :] = hidden_gaussian[:, patch_start_idx:, :] + rgb_feat

            point_hidden = self.point_decoder(hidden_upsampled, xpos=pos_upsampled)
            gaussian_hidden = self.gaussian_decoder(hidden_gaussian, xpos=pos_upsampled)
            camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast('cuda', enabled=False):
            out_h, out_w = patch_h * self.gaussians_per_axis, patch_w * self.gaussians_per_axis
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, patch_start_idx:]], (h, w)).reshape(b, v, out_h, out_w, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # gaussian
            gaussian_hidden = gaussian_hidden.float()
            gaussian_params = self.gaussian_head([gaussian_hidden[:, patch_start_idx:]], (h, w)).reshape(b, v, out_h, out_w, -1)

            gaussian_params = rearrange(gaussian_params, "b v h w d -> (b v) d h w").contiguous()

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, patch_start_idx:], patch_h, patch_w).reshape(b, v, 4, 4)  #  c2w

            # convert to the cooridinate system of the first view
            w2c_v1 = se3_inverse(camera_poses[:, 0])
            camera_poses = torch.einsum('bij, bnjk -> bnik', w2c_v1, camera_poses)

            pts_all = rearrange(local_points, "b v h w xyz -> (b v) xyz h w").contiguous()

        # judge if pts_all have 3 dimensions or 4 dimensions
        if pts_all.dim() == 4:
            pts_all = rearrange(pts_all, "(b v) d h w -> b v (h w) d", b=b, v=v)
        else:
            pts_all = rearrange(pts_all, "(b v) d l -> b v l d", b=b, v=v)

        # transform the pts into local coordinate system
        local_pts = pts_all.clone()  # b, v, l, 3

        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        depths = pts_all[..., -1].unsqueeze(-1)  # depth in a unified coordinate system

        if gaussian_params.dim() == 4:
            gaussians = rearrange(gaussian_params, "(b v) d h w -> b v (h w) d", b=b, v=v)
        else:
            gaussians = rearrange(gaussian_params, "(b v) d l -> b v l d", b=b, v=v)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        if self.pose_free:
            if self.training:
                prob_use_gt_pose = get_scheduled_sampling_epsilon(global_step,
                                                                  epsilon_end=self.cfg.gt_pose_final_sample_ratio,
                                                                  decay_start_step=self.cfg.gt_pose_sampling_decay_start_step,
                                                                  decay_end_step=self.cfg.gt_pose_sampling_decay_end_step, )
                if random.random() < prob_use_gt_pose:
                    c2w = context['extrinsics']
                else:
                    c2w = camera_poses
            else:
                # At inference, use GT extrinsics if explicitly requested
                # (predicted poses may not have converged, especially after fine-tuning)
                if context.get('use_gt_extrinsics', False):
                    c2w = context['extrinsics']
                else:
                    c2w = camera_poses
        else:
            c2w = context['extrinsics']

        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            depths,
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            extrinsics=rearrange(c2w, "b v i j -> b v () () () i j"),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            h, w = patch_h * self.gaussians_per_axis, patch_w * self.gaussians_per_axis
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).contiguous()
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump['local_pts'] = rearrange(
                local_pts.unsqueeze(-2), "b v (h w) srf xyz -> b v h w srf xyz", h=h, w=w
            )
            visualization_dump['pred_camera_poses'] = camera_poses.contiguous()
            visualization_dump['c2w'] = camera_poses.contiguous() if self.pose_free else context['extrinsics']
            visualization_dump['intrinsic_pred'] = intrinsic_pred

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            rearrange(gaussians.rotations,
                      "b v r srf spp d -> b (v r srf spp) d",
            ),
            rearrange(gaussians.scales,
                      "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim

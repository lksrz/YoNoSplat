from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossSilhouetteCfg:
    weight: float


@dataclass
class LossSilhouetteCfgWrapper:
    silhouette: LossSilhouetteCfg


class LossSilhouette(Loss[LossSilhouetteCfg, LossSilhouetteCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        views = batch["context"] if use_context else batch["target"]
        target_alpha = views.get("mask")
        if target_alpha is None or prediction.alpha is None:
            device = prediction.color.device
            return torch.tensor(0, dtype=torch.float32, device=device)

        target_alpha = target_alpha.to(prediction.alpha.device).clamp(0.0, 1.0)
        pred_alpha = prediction.alpha.clamp(1e-6, 1.0 - 1e-6)

        # Balance foreground and background so the model learns the silhouette
        # without collapsing object coverage on sparse crops.
        fg_fraction = target_alpha.mean().clamp_min(1e-3)
        bg_fraction = (1.0 - target_alpha).mean().clamp_min(1e-3)
        weights = target_alpha * (0.5 / fg_fraction) + (1.0 - target_alpha) * (
            0.5 / bg_fraction
        )
        weights = weights / weights.mean().clamp_min(1e-6)

        loss = F.binary_cross_entropy(pred_alpha, target_alpha, weight=weights)
        return self.cfg.weight * loss

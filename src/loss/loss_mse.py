from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        if use_context:
            delta = prediction.color - batch["context"]["image"]
            mask = batch["context"].get("mask", None)
        else:
            delta = prediction.color - batch["target"]["image"]
            mask = batch["target"].get("mask", None)
        sq_error = delta**2
        if mask is not None:
            # mask shape: [b, v, 1, H, W] — only compute loss on foreground
            mask = mask.to(sq_error.device)
            sq_error = sq_error * mask
            # Mean over foreground pixels only (avoid division by zero)
            num_fg = mask.sum().clamp_min(1.0)
            return self.cfg.weight * sq_error.sum() / num_fg
        return self.cfg.weight * sq_error.mean()

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .foreground_utils import get_supervised_images
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
            pred_image, target_image, mask = get_supervised_images(
                batch["context"],
                prediction.color,
            )
        else:
            pred_image, target_image, mask = get_supervised_images(
                batch["target"],
                prediction.color,
            )
        delta = pred_image - target_image
        sq_error = delta**2
        if mask is not None:
            # Mask shape: [b, v, 1, H, W]. With RGBA data, target_image is the
            # straight RGB foreground and both tensors are premultiplied by alpha.
            mask = mask.to(sq_error.device)
            # Mean over foreground pixels only (avoid division by zero)
            num_fg = mask.sum().clamp_min(1.0)
            return self.cfg.weight * sq_error.sum() / num_fg
        return self.cfg.weight * sq_error.mean()

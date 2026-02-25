from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossOpacityCfg:
    weight: float


@dataclass
class LossOpacityCfgWrapper:
    opacity: LossOpacityCfg


class LossOpacity(Loss[LossOpacityCfg, LossOpacityCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        return self.cfg.weight * gaussians.opacities.mean()

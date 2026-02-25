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
        else:
            delta = prediction.color - batch["target"]["image"]
        return self.cfg.weight * (delta**2).mean()

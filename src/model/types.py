from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    rotations: Optional[Float[Tensor, "batch gaussian 4"]]
    scales: Optional[Float[Tensor, "batch gaussian 3"]]

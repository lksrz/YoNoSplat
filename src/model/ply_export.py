from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def covariance_to_scaling_rotation(
    covariances: Float[Tensor, "gaussian 3 3"],
) -> tuple[
    Float[Tensor, "gaussian 3"],
    Float[Tensor, "gaussian 4"],
]:
    eigvals, eigvecs = torch.linalg.eigh(covariances.detach().cpu())
    order = torch.argsort(eigvals, dim=-1, descending=True)
    eigvals = torch.gather(eigvals, -1, order)
    eigvecs = torch.gather(eigvecs, -1, order.unsqueeze(-2).expand(-1, 3, 3))

    det = torch.det(eigvecs)
    eigvecs[det < 0, :, -1] *= -1

    scales = eigvals.clamp_min(1e-12).sqrt()
    rotations = torch.from_numpy(R.from_matrix(eigvecs.numpy()).as_quat()).to(
        dtype=covariances.dtype
    )
    return scales, rotations


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    covariances: Float[Tensor, "gaussian 3 3"] | None = None,
):
    # Move everything to CPU first for consistency.
    means = means.detach().cpu()
    scales = scales.detach().cpu()
    rotations = rotations.detach().cpu()
    harmonics = harmonics.detach().cpu()
    opacities = opacities.detach().cpu()

    if covariances is not None:
        scales, rotations = covariance_to_scaling_rotation(covariances.detach().cpu())

    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.numpy(),
        torch.zeros_like(means).numpy(),
        f_dc.contiguous().numpy(),
        f_rest.contiguous().numpy(),
        torch.logit(opacities.clamp(1e-6, 1 - 1e-6))[..., None].numpy(),
        scales.log().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

import torch

from ..types import Gaussians


def prune_gaussians(
    gaussians: Gaussians,
    opacity_threshold: float,
    prune_ratio: float,
    random_keep_ratio: float,
    inference: bool = False,
) -> Gaussians:
    """Prune the Gaussians to only include those that are visible in the image."""
    means = gaussians.means  # (B, G, 3)
    opacities = gaussians.opacities  # (B, G)

    if means.shape[0] > 1:
        assert not inference, "Inference mode is not supported when bs > 1."

    if inference and opacity_threshold > 0:
        # Inference mode: prune based on opacity threshold
        gaussian_mask = opacities > opacity_threshold  # (B, G)

        num_gaussians = means.shape[1]
        num_pruned = num_gaussians - gaussian_mask.sum()
        # print(f"Pruned {num_pruned} gaussians out of {num_gaussians} ({num_pruned / num_gaussians:.2%})")

        def trim(element, mask):
            return element[mask].unsqueeze(0)

        gaussians = Gaussians(
            means=trim(gaussians.means, gaussian_mask),
            covariances=trim(gaussians.covariances, gaussian_mask),
            harmonics=trim(gaussians.harmonics, gaussian_mask),
            opacities=trim(gaussians.opacities, gaussian_mask),
            rotations=trim(gaussians.rotations, gaussian_mask),
            scales=trim(gaussians.scales, gaussian_mask),
        )

        return gaussians

    if prune_ratio > 0:
        # Training mode: prune based on opacity and random sampling (fixed ratio)
        num_gaussians = means.shape[1]

        keep_ratio = 1 - prune_ratio
        random_keep_ratio = keep_ratio * random_keep_ratio
        keep_ratio = keep_ratio - random_keep_ratio
        num_keep = int(num_gaussians * keep_ratio)
        num_keep_random = int(num_gaussians * random_keep_ratio)
        # rank by opacity
        idx_sort = opacities.argsort(dim=1, descending=True)
        keep_idx = idx_sort[:, :num_keep]
        if num_keep_random > 0:
            rest_idx = idx_sort[:, num_keep:]
            random_idx = rest_idx[:, torch.randperm(rest_idx.shape[1])[:num_keep_random]]
            keep_idx = torch.cat([keep_idx, random_idx], dim=1)

        return Gaussians(
            means=gaussians.means.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, gaussians.means.shape[-1])),
            covariances=gaussians.covariances.gather(1, keep_idx[..., None, None].expand(-1, -1, gaussians.covariances.shape[-2], gaussians.covariances.shape[-1])),
            harmonics=gaussians.harmonics.gather(1, keep_idx[..., None, None].expand(-1, -1, gaussians.harmonics.shape[-2], gaussians.harmonics.shape[-1])),
            opacities=gaussians.opacities.gather(1, keep_idx),
            rotations=gaussians.rotations.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, gaussians.rotations.shape[-1])),
            scales=gaussians.scales.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, gaussians.scales.shape[-1])),
        )

    return gaussians

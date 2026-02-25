import torch
from jaxtyping import Float
from torch import Tensor


def compute_pose_norm_scale(
    context_extrinsics: Float[Tensor, "view 4 4"],
    method: str,
) -> float | Tensor:
    """Compute the scale factor for pose normalization.

    Args:
        context_extrinsics: Camera-to-world extrinsic matrices for context views.
        method: One of "none", "start_end", "max_pairwise_d", "max_view1_d",
                "mean_pairwise_d", "max_trans".

    Returns:
        The computed scale value.
    """
    if method == "start_end":
        a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
        scale = (a - b).norm()
    elif method == "max_pairwise_d":
        scale = 0
        for i in range(context_extrinsics.shape[0]):
            for j in range(i + 1, context_extrinsics.shape[0]):
                a, b = context_extrinsics[i, :3, 3], context_extrinsics[j, :3, 3]
                scale = max(scale, (a - b).norm())
    elif method == "mean_pairwise_d":
        # For c2w matrices, camera position is just the translation part
        camera_positions = context_extrinsics[:, :3, 3]  # Shape: (V, 3)

        # Calculate pairwise distances using broadcasting
        positions_i = camera_positions.unsqueeze(1)  # Shape: (V, 1, 3)
        positions_j = camera_positions.unsqueeze(0)  # Shape: (1, V, 3)

        # Compute pairwise distance matrix
        distance_matrix = torch.norm(positions_i - positions_j, dim=2)  # Shape: (V, V)

        # Extract upper triangular part (excluding diagonal) to get unique pairs
        mask = torch.triu(
            torch.ones(distance_matrix.shape[0], distance_matrix.shape[1], dtype=torch.bool),
            diagonal=1,
        )
        pairwise_distances = distance_matrix[mask]

        # Calculate average distance
        scale = torch.mean(pairwise_distances)
    elif method == "max_trans":
        scale = torch.max(torch.abs(context_extrinsics[:, :3, 3]))
        scale = torch.norm(scale)  # re-scale the scene to a fixed scale
    elif method == "max_view1_d":
        view1 = context_extrinsics[0:1, :3, 3]
        view_remaining = context_extrinsics[1:, :3, 3]
        scale = (view1 - view_remaining).norm(dim=-1).max()
    elif method == "none":
        scale = 1.0
    else:
        raise ValueError(f"Unknown pose norm method {method}")

    return scale

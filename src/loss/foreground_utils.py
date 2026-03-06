from ..dataset.types import BatchedViews


def get_supervised_images(views: BatchedViews, prediction_color):
    """Return images for foreground-only supervision when alpha is available."""
    target = views["image"].to(prediction_color.device)
    mask = views.get("mask")
    if mask is None:
        return prediction_color, target, None

    mask = mask.to(prediction_color.device).clamp(0.0, 1.0)
    foreground = views.get("foreground")
    if foreground is not None:
        target = foreground.to(prediction_color.device)

    return prediction_color * mask, target * mask, mask

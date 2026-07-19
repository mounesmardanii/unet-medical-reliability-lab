"""Evaluation metrics for binary medical image segmentation."""

from __future__ import annotations

import torch


@torch.no_grad()
def binary_segmentation_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Compute mean Dice and IoU scores from segmentation logits."""

    if logits.shape != targets.shape:
        raise ValueError(
            "Logits and targets must have identical shapes: "
            f"{logits.shape} versus {targets.shape}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between zero and one."
        )

    if epsilon <= 0:
        raise ValueError(
            "Epsilon must be greater than zero."
        )

    targets = targets.to(
        dtype=logits.dtype,
    )

    # Convert the model's raw logits into probabilities.
    probabilities = torch.sigmoid(logits)

    # Convert probabilities into a binary predicted mask.
    predictions = (
        probabilities >= threshold
    ).to(dtype=logits.dtype)

    # Ensure that the ground-truth mask is also binary.
    binary_targets = (
        targets >= 0.5
    ).to(dtype=logits.dtype)

    batch_size = logits.shape[0]

    # Flatten each image and mask separately.
    predictions = predictions.reshape(
        batch_size,
        -1,
    )

    binary_targets = binary_targets.reshape(
        batch_size,
        -1,
    )

    intersection = (
        predictions * binary_targets
    ).sum(dim=1)

    prediction_size = predictions.sum(dim=1)
    target_size = binary_targets.sum(dim=1)

    dice_scores = (
        2.0 * intersection + epsilon
    ) / (
        prediction_size
        + target_size
        + epsilon
    )

    union = (
        prediction_size
        + target_size
        - intersection
    )

    iou_scores = (
        intersection + epsilon
    ) / (
        union + epsilon
    )

    return {
        "dice": dice_scores.mean(),
        "iou": iou_scores.mean(),
    }
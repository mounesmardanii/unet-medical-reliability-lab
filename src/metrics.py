"""Evaluation metrics for binary medical image segmentation."""

from __future__ import annotations

import torch


@torch.no_grad()
def binary_segmentation_statistics_per_image_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Compute per-image segmentation metrics and confusion counts."""

    if logits.shape != targets.shape:
        raise ValueError(
            "Logits and targets must have identical shapes: "
            f"{logits.shape} versus {targets.shape}"
        )

    if logits.ndim < 2:
        raise ValueError(
            "Logits and targets must include a batch dimension."
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

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).to(dtype=logits.dtype)

    binary_targets = (
        targets >= 0.5
    ).to(dtype=logits.dtype)

    batch_size = logits.shape[0]

    predictions = predictions.reshape(
        batch_size,
        -1,
    )

    binary_targets = binary_targets.reshape(
        batch_size,
        -1,
    )

    true_positive = (
        predictions * binary_targets
    ).sum(dim=1)

    false_positive = (
        predictions * (1.0 - binary_targets)
    ).sum(dim=1)

    false_negative = (
        (1.0 - predictions) * binary_targets
    ).sum(dim=1)

    true_negative = (
        (1.0 - predictions)
        * (1.0 - binary_targets)
    ).sum(dim=1)

    dice_scores = (
        2.0 * true_positive + epsilon
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    iou_scores = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    precision_scores = true_positive / (
        true_positive
        + false_positive
        + epsilon
    )

    sensitivity_scores = true_positive / (
        true_positive
        + false_negative
        + epsilon
    )

    specificity_scores = true_negative / (
        true_negative
        + false_positive
        + epsilon
    )

    total_pixels = (
        true_positive
        + false_positive
        + false_negative
        + true_negative
    )

    predicted_positive_fraction = (
        true_positive + false_positive
    ) / (
        total_pixels + epsilon
    )

    target_positive_fraction = (
        true_positive + false_negative
    ) / (
        total_pixels + epsilon
    )

    return {
        "dice": dice_scores,
        "iou": iou_scores,
        "precision": precision_scores,
        "sensitivity": sensitivity_scores,
        "specificity": specificity_scores,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "predicted_positive_fraction": (
            predicted_positive_fraction
        ),
        "target_positive_fraction": (
            target_positive_fraction
        ),
    }


@torch.no_grad()
def binary_segmentation_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Compute mean Dice and IoU scores from segmentation logits."""

    statistics = (
        binary_segmentation_statistics_per_image_from_logits(
            logits=logits,
            targets=targets,
            threshold=threshold,
            epsilon=epsilon,
        )
    )

    return {
        "dice": statistics["dice"].mean(),
        "iou": statistics["iou"].mean(),
    }
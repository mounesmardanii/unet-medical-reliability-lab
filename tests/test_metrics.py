"""Tests for binary segmentation evaluation metrics."""

import pytest
import torch

from src.metrics import (
    binary_segmentation_metrics_from_logits,
)


def test_metrics_reward_perfect_predictions() -> None:
    """Perfect predictions should produce Dice and IoU near one."""

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ]
    )

    logits = torch.where(
        targets == 1.0,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )

    metrics = binary_segmentation_metrics_from_logits(
        logits,
        targets,
        threshold=0.5,
    )

    assert metrics["dice"].item() > 0.999
    assert metrics["iou"].item() > 0.999


def test_metrics_penalize_wrong_predictions() -> None:
    """Completely wrong predictions should produce scores near zero."""

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ]
    )

    logits = torch.where(
        targets == 1.0,
        torch.tensor(-20.0),
        torch.tensor(20.0),
    )

    metrics = binary_segmentation_metrics_from_logits(
        logits,
        targets,
        threshold=0.5,
    )

    assert metrics["dice"].item() < 0.001
    assert metrics["iou"].item() < 0.001


def test_metrics_compute_per_image_mean() -> None:
    """Batch metrics should average image-level scores."""

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
        ]
    )

    logits = torch.tensor(
        [
            [
                [
                    [20.0, 20.0],
                    [-20.0, -20.0],
                ]
            ],
            [
                [
                    [-20.0, -20.0],
                    [20.0, 20.0],
                ]
            ],
        ]
    )

    metrics = binary_segmentation_metrics_from_logits(
        logits,
        targets,
        threshold=0.5,
    )

    # The first image scores approximately 1,
    # while the second image scores approximately 0.
    # Their image-level mean should therefore be about 0.5.
    assert metrics["dice"].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert metrics["iou"].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )


def test_metrics_reject_shape_mismatch() -> None:
    """Metrics should reject incompatible tensor shapes."""

    logits = torch.zeros(
        2,
        1,
        8,
        8,
    )

    targets = torch.zeros(
        2,
        1,
        4,
        4,
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        binary_segmentation_metrics_from_logits(
            logits,
            targets,
        )


def test_metrics_reject_invalid_threshold() -> None:
    """Threshold values outside zero and one should be rejected."""

    logits = torch.zeros(
        1,
        1,
        4,
        4,
    )

    targets = torch.zeros_like(logits)

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        binary_segmentation_metrics_from_logits(
            logits,
            targets,
            threshold=1.5,
        )
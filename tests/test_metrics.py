"""Tests for binary segmentation evaluation metrics."""

import pytest
import torch

from src.metrics import (
    binary_segmentation_metrics_from_logits,
    binary_segmentation_statistics_per_image_from_logits,
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
        logits=logits,
        targets=targets,
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
        logits=logits,
        targets=targets,
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
        logits=logits,
        targets=targets,
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
            logits=logits,
            targets=targets,
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
            logits=logits,
            targets=targets,
            threshold=1.5,
        )


def test_per_image_statistics_reward_perfect_predictions() -> None:
    """Perfect masks should produce ideal per-image statistics."""

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ]
        ]
    )

    logits = torch.where(
        targets == 1.0,
        torch.tensor(20.0),
        torch.tensor(-20.0),
    )

    statistics = (
        binary_segmentation_statistics_per_image_from_logits(
            logits=logits,
            targets=targets,
            threshold=0.5,
        )
    )

    expected_keys = {
        "dice",
        "iou",
        "precision",
        "sensitivity",
        "specificity",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "predicted_positive_fraction",
        "target_positive_fraction",
    }

    assert set(statistics) == expected_keys

    for metric_name in [
        "dice",
        "iou",
        "precision",
        "sensitivity",
        "specificity",
    ]:
        assert statistics[metric_name].shape == (1,)

        assert statistics[
            metric_name
        ].item() == pytest.approx(
            1.0,
            abs=1e-6,
        )

    assert statistics[
        "true_positive"
    ].item() == pytest.approx(
        2.0,
    )

    assert statistics[
        "false_positive"
    ].item() == pytest.approx(
        0.0,
    )

    assert statistics[
        "false_negative"
    ].item() == pytest.approx(
        0.0,
    )

    assert statistics[
        "true_negative"
    ].item() == pytest.approx(
        2.0,
    )

    assert statistics[
        "predicted_positive_fraction"
    ].item() == pytest.approx(
        0.5,
    )

    assert statistics[
        "target_positive_fraction"
    ].item() == pytest.approx(
        0.5,
    )


def test_per_image_statistics_match_known_confusion_counts() -> None:
    """Statistics should match a known pixel confusion matrix."""

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ]
        ]
    )

    logits = torch.tensor(
        [
            [
                [
                    [20.0, -20.0],
                    [20.0, -20.0],
                ]
            ]
        ]
    )

    statistics = (
        binary_segmentation_statistics_per_image_from_logits(
            logits=logits,
            targets=targets,
            threshold=0.5,
        )
    )

    assert statistics[
        "true_positive"
    ].item() == pytest.approx(
        1.0,
    )

    assert statistics[
        "false_positive"
    ].item() == pytest.approx(
        1.0,
    )

    assert statistics[
        "false_negative"
    ].item() == pytest.approx(
        1.0,
    )

    assert statistics[
        "true_negative"
    ].item() == pytest.approx(
        1.0,
    )

    assert statistics["dice"].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert statistics["iou"].item() == pytest.approx(
        1.0 / 3.0,
        abs=1e-6,
    )

    assert statistics[
        "precision"
    ].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert statistics[
        "sensitivity"
    ].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert statistics[
        "specificity"
    ].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert statistics[
        "predicted_positive_fraction"
    ].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert statistics[
        "target_positive_fraction"
    ].item() == pytest.approx(
        0.5,
        abs=1e-6,
    )


def test_per_image_statistics_preserve_batch_dimension() -> None:
    """Each image should receive its own metric values."""

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

    statistics = (
        binary_segmentation_statistics_per_image_from_logits(
            logits=logits,
            targets=targets,
            threshold=0.5,
        )
    )

    for values in statistics.values():
        assert values.shape == (2,)

    assert statistics["dice"][0].item() == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert statistics["dice"][1].item() == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert statistics["iou"][0].item() == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert statistics["iou"][1].item() == pytest.approx(
        0.0,
        abs=1e-6,
    )
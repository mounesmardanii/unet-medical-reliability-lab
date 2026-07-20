"""Tests for BUSI final-evaluation utilities."""

from __future__ import annotations

import pytest
import torch

from scripts.evaluate_busi_test import (
    compute_micro_metrics,
    summarize_values,
    validate_threshold_checkpoint_consistency,
)


def test_summarize_values_uses_standard_even_sample_median() -> None:
    """Median of an even-sized sample should average middle values."""

    values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float32,
    )

    summary = summarize_values(
        values=values,
        bootstrap_iterations=1000,
        confidence_level=0.95,
        seed=42,
    )

    assert summary["sample_count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["median"] == pytest.approx(2.5)


def test_summarize_values_bootstrap_is_reproducible() -> None:
    """The same seed should produce identical confidence intervals."""

    values = torch.tensor(
        [0.2, 0.4, 0.6, 0.8],
        dtype=torch.float32,
    )

    first_summary = summarize_values(
        values=values,
        bootstrap_iterations=1000,
        confidence_level=0.95,
        seed=42,
    )

    second_summary = summarize_values(
        values=values,
        bootstrap_iterations=1000,
        confidence_level=0.95,
        seed=42,
    )

    assert first_summary["ci_lower"] == pytest.approx(
        second_summary["ci_lower"]
    )

    assert first_summary["ci_upper"] == pytest.approx(
        second_summary["ci_upper"]
    )


def test_compute_micro_metrics_matches_known_counts() -> None:
    """Pooled metrics should be calculated from summed pixel counts."""

    statistics = {
        "true_positive": torch.tensor([3.0, 1.0]),
        "false_positive": torch.tensor([1.0, 1.0]),
        "false_negative": torch.tensor([1.0, 3.0]),
        "true_negative": torch.tensor([5.0, 5.0]),
    }

    selector = torch.tensor(
        [True, True],
        dtype=torch.bool,
    )

    metrics = compute_micro_metrics(
        statistics=statistics,
        selector=selector,
    )

    assert metrics["true_positive"] == pytest.approx(4.0)
    assert metrics["false_positive"] == pytest.approx(2.0)
    assert metrics["false_negative"] == pytest.approx(4.0)
    assert metrics["true_negative"] == pytest.approx(10.0)

    assert metrics["dice"] == pytest.approx(
        8.0 / 14.0
    )

    assert metrics["iou"] == pytest.approx(
        4.0 / 10.0
    )

    assert metrics["precision"] == pytest.approx(
        4.0 / 6.0
    )

    assert metrics["sensitivity"] == pytest.approx(
        4.0 / 8.0
    )

    assert metrics["specificity"] == pytest.approx(
        10.0 / 12.0
    )


def test_checkpoint_consistency_accepts_matching_metadata() -> None:
    """Matching checkpoint metadata should pass without error."""

    threshold_results = {
        "checkpoint_epoch": 35,
        "checkpoint_validation_dice": 0.8133323978870473,
    }

    checkpoint = {
        "epoch": 35,
        "validation_dice": 0.8133323978870473,
    }

    validate_threshold_checkpoint_consistency(
        threshold_results=threshold_results,
        checkpoint=checkpoint,
    )


def test_checkpoint_consistency_rejects_different_epoch() -> None:
    """Threshold and model must refer to the same checkpoint epoch."""

    threshold_results = {
        "checkpoint_epoch": 35,
        "checkpoint_validation_dice": 0.81,
    }

    checkpoint = {
        "epoch": 34,
        "validation_dice": 0.81,
    }

    with pytest.raises(
        ValueError,
        match="checkpoint epoch",
    ):
        validate_threshold_checkpoint_consistency(
            threshold_results=threshold_results,
            checkpoint=checkpoint,
        )


def test_checkpoint_consistency_rejects_different_validation_dice() -> None:
    """Threshold and model validation scores must match."""

    threshold_results = {
        "checkpoint_epoch": 35,
        "checkpoint_validation_dice": 0.81,
    }

    checkpoint = {
        "epoch": 35,
        "validation_dice": 0.82,
    }

    with pytest.raises(
        ValueError,
        match="validation Dice",
    ):
        validate_threshold_checkpoint_consistency(
            threshold_results=threshold_results,
            checkpoint=checkpoint,
        )
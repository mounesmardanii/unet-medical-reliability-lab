"""Tests for segmentation reliability statistics."""

from __future__ import annotations

import pytest
import torch

from src.reliability import (
    bootstrap_independent_mean_difference,
)


def test_bootstrap_reports_observed_mean_difference() -> None:
    """Reported difference must equal first mean minus second mean."""

    first_group = torch.tensor(
        [0.8, 1.0],
        dtype=torch.float32,
    )

    second_group = torch.tensor(
        [0.2, 0.6],
        dtype=torch.float32,
    )

    result = bootstrap_independent_mean_difference(
        first_group=first_group,
        second_group=second_group,
        bootstrap_iterations=1000,
        confidence_level=0.95,
        seed=42,
    )

    assert result["first_group_sample_count"] == 2
    assert result["second_group_sample_count"] == 2

    assert result["first_group_mean"] == pytest.approx(
        0.9
    )

    assert result["second_group_mean"] == pytest.approx(
        0.4
    )

    assert result["mean_difference"] == pytest.approx(
        0.5
    )


def test_bootstrap_is_reproducible_with_same_seed() -> None:
    """Identical seeds must produce identical confidence intervals."""

    first_group = torch.tensor(
        [0.7, 0.8, 0.9, 1.0],
        dtype=torch.float32,
    )

    second_group = torch.tensor(
        [0.3, 0.4, 0.5, 0.6],
        dtype=torch.float32,
    )

    first_result = bootstrap_independent_mean_difference(
        first_group=first_group,
        second_group=second_group,
        bootstrap_iterations=1000,
        seed=42,
    )

    second_result = bootstrap_independent_mean_difference(
        first_group=first_group,
        second_group=second_group,
        bootstrap_iterations=1000,
        seed=42,
    )

    assert first_result["ci_lower"] == pytest.approx(
        second_result["ci_lower"]
    )

    assert first_result["ci_upper"] == pytest.approx(
        second_result["ci_upper"]
    )


def test_binary_values_represent_proportion_difference() -> None:
    """For binary values, the mean difference is a rate difference."""

    first_group = torch.tensor(
        [1.0, 1.0, 0.0, 0.0],
    )

    second_group = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
    )

    result = bootstrap_independent_mean_difference(
        first_group=first_group,
        second_group=second_group,
        bootstrap_iterations=1000,
        seed=42,
    )

    assert result["first_group_mean"] == pytest.approx(
        0.50
    )

    assert result["second_group_mean"] == pytest.approx(
        0.25
    )

    assert result["mean_difference"] == pytest.approx(
        0.25
    )


def test_bootstrap_rejects_empty_group() -> None:
    """Both groups must contain at least one value."""

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        bootstrap_independent_mean_difference(
            first_group=torch.tensor([]),
            second_group=torch.tensor([0.5]),
        )


def test_bootstrap_rejects_multidimensional_values() -> None:
    """Each group must be represented by one value per image."""

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        bootstrap_independent_mean_difference(
            first_group=torch.tensor(
                [[0.5, 0.6]]
            ),
            second_group=torch.tensor(
                [0.4, 0.5]
            ),
        )


def test_bootstrap_rejects_nonfinite_values() -> None:
    """NaN and infinite metric values must not enter the analysis."""

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        bootstrap_independent_mean_difference(
            first_group=torch.tensor(
                [0.5, float("nan")]
            ),
            second_group=torch.tensor(
                [0.4, 0.6]
            ),
        )


def test_bootstrap_rejects_invalid_iteration_count() -> None:
    """At least one bootstrap iteration is required."""

    with pytest.raises(
        ValueError,
        match="iterations",
    ):
        bootstrap_independent_mean_difference(
            first_group=torch.tensor([0.5]),
            second_group=torch.tensor([0.4]),
            bootstrap_iterations=0,
        )


def test_bootstrap_rejects_invalid_confidence_level() -> None:
    """Confidence level must be strictly between zero and one."""

    with pytest.raises(
        ValueError,
        match="Confidence level",
    ):
        bootstrap_independent_mean_difference(
            first_group=torch.tensor([0.5]),
            second_group=torch.tensor([0.4]),
            confidence_level=1.0,
        )
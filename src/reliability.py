"""Statistical utilities for segmentation reliability analysis."""

from __future__ import annotations

import torch


def _validate_metric_values(
    values: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Convert metric values to a validated one-dimensional tensor."""

    values = values.detach().cpu().to(
        dtype=torch.float64
    )

    if values.ndim != 1:
        raise ValueError(
            f"{name} values must be one-dimensional."
        )

    if values.numel() == 0:
        raise ValueError(
            f"{name} values cannot be empty."
        )

    if not torch.isfinite(values).all():
        raise ValueError(
            f"{name} values must all be finite."
        )

    return values


def bootstrap_independent_mean_difference(
    first_group: torch.Tensor,
    second_group: torch.Tensor,
    bootstrap_iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, int | float]:
    """
    Estimate the difference between two independent group means.

    The reported difference is:

        first group mean - second group mean

    For binary values, the same calculation represents a
    difference between two proportions.
    """

    first_group = _validate_metric_values(
        values=first_group,
        name="First group",
    )

    second_group = _validate_metric_values(
        values=second_group,
        name="Second group",
    )

    if bootstrap_iterations <= 0:
        raise ValueError(
            "Bootstrap iterations must be greater than zero."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "Confidence level must be between zero and one."
        )

    first_count = first_group.numel()
    second_count = second_group.numel()

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    first_indices = torch.randint(
        low=0,
        high=first_count,
        size=(
            bootstrap_iterations,
            first_count,
        ),
        generator=generator,
    )

    second_indices = torch.randint(
        low=0,
        high=second_count,
        size=(
            bootstrap_iterations,
            second_count,
        ),
        generator=generator,
    )

    bootstrap_differences = (
        first_group[first_indices].mean(dim=1)
        - second_group[second_indices].mean(dim=1)
    )

    alpha = (
        1.0 - confidence_level
    ) / 2.0

    confidence_interval_lower = torch.quantile(
        bootstrap_differences,
        alpha,
    )

    confidence_interval_upper = torch.quantile(
        bootstrap_differences,
        1.0 - alpha,
    )

    observed_difference = (
        first_group.mean()
        - second_group.mean()
    )

    return {
        "first_group_sample_count": first_count,
        "second_group_sample_count": second_count,
        "first_group_mean": float(
            first_group.mean()
        ),
        "second_group_mean": float(
            second_group.mean()
        ),
        "mean_difference": float(
            observed_difference
        ),
        "confidence_level": confidence_level,
        "ci_lower": float(
            confidence_interval_lower
        ),
        "ci_upper": float(
            confidence_interval_upper
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
    }
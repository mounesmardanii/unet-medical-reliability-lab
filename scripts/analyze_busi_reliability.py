"""Analyze subgroup reliability gaps from saved BUSI test results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from src.reliability import (
    bootstrap_independent_mean_difference,
)


REPORT_METRICS = (
    "dice",
    "iou",
    "precision",
    "sensitivity",
    "specificity",
)

REQUIRED_COLUMNS = {
    "filename",
    "class_name",
    *REPORT_METRICS,
}

RATE_DEFINITIONS = (
    {
        "name": "dice_below_0_05",
        "description": "Near-complete segmentation failure",
        "operator": "less_than",
        "threshold": 0.05,
    },
    {
        "name": "dice_below_0_50",
        "description": "Poor segmentation performance",
        "operator": "less_than",
        "threshold": 0.50,
    },
    {
        "name": "dice_below_0_70",
        "description": "Low or unreliable segmentation performance",
        "operator": "less_than",
        "threshold": 0.70,
    },
    {
        "name": "dice_at_least_0_90",
        "description": "Excellent segmentation performance",
        "operator": "greater_than_or_equal",
        "threshold": 0.90,
    },
)


def parse_arguments() -> argparse.Namespace:
    """Read reliability-analysis settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare benign and malignant BUSI test "
            "performance using image-level bootstrap intervals."
        )
    )

    parser.add_argument(
        "--per-image-results-path",
        type=Path,
        default=Path(
            "artifacts/busi_baseline_128_seed42/"
            "test_evaluation/per_image_results.csv"
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "artifacts/busi_baseline_128_seed42/"
            "test_evaluation/reliability_analysis.json"
        ),
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Reject missing files and invalid statistical settings."""

    if not arguments.per_image_results_path.is_file():
        raise FileNotFoundError(
            "Per-image results file was not found: "
            f"{arguments.per_image_results_path}"
        )

    if arguments.bootstrap_iterations <= 0:
        raise ValueError(
            "Bootstrap iterations must be greater than zero."
        )

    if not 0.0 < arguments.confidence_level < 1.0:
        raise ValueError(
            "Confidence level must be between zero and one."
        )


def load_per_image_results(
    input_path: Path,
) -> list[dict[str, str | float]]:
    """Load and validate saved image-level test metrics."""

    rows: list[dict[str, str | float]] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                "Per-image results CSV has no header."
            )

        missing_columns = (
            REQUIRED_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Per-image results CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            class_name = row["class_name"].strip()

            if class_name not in {
                "benign",
                "malignant",
            }:
                raise ValueError(
                    "Unexpected class name at CSV line "
                    f"{line_number}: {class_name!r}"
                )

            parsed_row: dict[str, str | float] = {
                "filename": row["filename"].strip(),
                "class_name": class_name,
            }

            for metric_name in REPORT_METRICS:
                try:
                    value = float(
                        row[metric_name]
                    )
                except (
                    TypeError,
                    ValueError,
                ) as error:
                    raise ValueError(
                        "Invalid numeric value for "
                        f"{metric_name!r} at CSV line "
                        f"{line_number}."
                    ) from error

                if not math.isfinite(value):
                    raise ValueError(
                        "Non-finite value for "
                        f"{metric_name!r} at CSV line "
                        f"{line_number}."
                    )

                parsed_row[metric_name] = value

            rows.append(parsed_row)

    if not rows:
        raise ValueError(
            "Per-image results CSV contains no samples."
        )

    observed_classes = {
        str(row["class_name"])
        for row in rows
    }

    if observed_classes != {
        "benign",
        "malignant",
    }:
        raise ValueError(
            "Both benign and malignant samples are required."
        )

    return rows


def get_metric_values(
    rows: list[dict[str, str | float]],
    class_name: str,
    metric_name: str,
) -> torch.Tensor:
    """Extract one metric for one diagnostic class."""

    values = [
        float(row[metric_name])
        for row in rows
        if row["class_name"] == class_name
    ]

    if not values:
        raise ValueError(
            f"No samples were found for class "
            f"{class_name!r}."
        )

    return torch.tensor(
        values,
        dtype=torch.float64,
    )


def build_rate_indicator(
    dice_values: torch.Tensor,
    operator: str,
    threshold: float,
) -> torch.Tensor:
    """Convert Dice values into binary event indicators."""

    if operator == "less_than":
        indicators = dice_values < threshold
    elif operator == "greater_than_or_equal":
        indicators = dice_values >= threshold
    else:
        raise ValueError(
            f"Unsupported rate operator: {operator!r}"
        )

    return indicators.to(
        dtype=torch.float64
    )


def format_difference_result(
    result: dict[str, int | float],
) -> dict[str, int | float | bool]:
    """Give bootstrap output meaningful subgroup labels."""

    ci_lower = float(
        result["ci_lower"]
    )

    ci_upper = float(
        result["ci_upper"]
    )

    return {
        "benign_sample_count": int(
            result["first_group_sample_count"]
        ),
        "malignant_sample_count": int(
            result["second_group_sample_count"]
        ),
        "benign_mean": float(
            result["first_group_mean"]
        ),
        "malignant_mean": float(
            result["second_group_mean"]
        ),
        "difference_benign_minus_malignant": float(
            result["mean_difference"]
        ),
        "confidence_level": float(
            result["confidence_level"]
        ),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_excludes_zero": (
            ci_lower > 0.0
            or ci_upper < 0.0
        ),
        "bootstrap_iterations": int(
            result["bootstrap_iterations"]
        ),
        "seed": int(
            result["seed"]
        ),
    }


def analyze_metric_differences(
    rows: list[dict[str, str | float]],
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, int | float | bool]]:
    """Compare benign and malignant means for each metric."""

    results = {}

    for metric_index, metric_name in enumerate(
        REPORT_METRICS
    ):
        benign_values = get_metric_values(
            rows=rows,
            class_name="benign",
            metric_name=metric_name,
        )

        malignant_values = get_metric_values(
            rows=rows,
            class_name="malignant",
            metric_name=metric_name,
        )

        bootstrap_result = (
            bootstrap_independent_mean_difference(
                first_group=benign_values,
                second_group=malignant_values,
                bootstrap_iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                seed=seed + metric_index,
            )
        )

        results[metric_name] = (
            format_difference_result(
                bootstrap_result
            )
        )

    return results


def analyze_rate_differences(
    rows: list[dict[str, str | float]],
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Compare subgroup failure and success rates."""

    benign_dice = get_metric_values(
        rows=rows,
        class_name="benign",
        metric_name="dice",
    )

    malignant_dice = get_metric_values(
        rows=rows,
        class_name="malignant",
        metric_name="dice",
    )

    results: dict[str, dict[str, object]] = {}

    for rate_index, definition in enumerate(
        RATE_DEFINITIONS
    ):
        threshold = float(
            definition["threshold"]
        )

        operator = str(
            definition["operator"]
        )

        benign_indicators = build_rate_indicator(
            dice_values=benign_dice,
            operator=operator,
            threshold=threshold,
        )

        malignant_indicators = build_rate_indicator(
            dice_values=malignant_dice,
            operator=operator,
            threshold=threshold,
        )

        bootstrap_result = (
            bootstrap_independent_mean_difference(
                first_group=benign_indicators,
                second_group=malignant_indicators,
                bootstrap_iterations=bootstrap_iterations,
                confidence_level=confidence_level,
                seed=seed + 100 + rate_index,
            )
        )

        formatted_result = (
            format_difference_result(
                bootstrap_result
            )
        )

        formatted_result.update(
            {
                "description": (
                    definition["description"]
                ),
                "operator": operator,
                "threshold": threshold,
                "benign_event_count": int(
                    benign_indicators.sum().item()
                ),
                "malignant_event_count": int(
                    malignant_indicators.sum().item()
                ),
            }
        )

        results[str(definition["name"])] = (
            formatted_result
        )

    return results


def save_json(
    content: dict[str, object],
    output_path: Path,
) -> None:
    """Save the reliability report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            content,
            output_file,
            indent=2,
        )


def main() -> None:
    """Run subgroup reliability analysis."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    rows = load_per_image_results(
        arguments.per_image_results_path
    )

    metric_differences = (
        analyze_metric_differences(
            rows=rows,
            bootstrap_iterations=(
                arguments.bootstrap_iterations
            ),
            confidence_level=(
                arguments.confidence_level
            ),
            seed=arguments.seed,
        )
    )

    rate_differences = (
        analyze_rate_differences(
            rows=rows,
            bootstrap_iterations=(
                arguments.bootstrap_iterations
            ),
            confidence_level=(
                arguments.confidence_level
            ),
            seed=arguments.seed,
        )
    )

    report = {
        "analysis_source": str(
            arguments.per_image_results_path
        ),
        "comparison_direction": (
            "benign_minus_malignant"
        ),
        "interpretation": {
            "performance_metrics": (
                "Positive differences indicate higher "
                "performance for benign samples."
            ),
            "failure_rates": (
                "Negative differences indicate a higher "
                "failure rate for malignant samples."
            ),
            "excellent_performance_rate": (
                "Positive differences indicate that excellent "
                "performance is more frequent for benign samples."
            ),
            "confidence_interval": (
                "A percentile bootstrap interval excluding zero "
                "provides evidence of a subgroup difference."
            ),
        },
        "bootstrap": {
            "iterations": (
                arguments.bootstrap_iterations
            ),
            "confidence_level": (
                arguments.confidence_level
            ),
            "seed": arguments.seed,
            "resampling_unit": "image",
            "groups_resampled_independently": True,
        },
        "sample_counts": {
            "benign": sum(
                row["class_name"] == "benign"
                for row in rows
            ),
            "malignant": sum(
                row["class_name"] == "malignant"
                for row in rows
            ),
            "total": len(rows),
        },
        "metric_mean_differences": (
            metric_differences
        ),
        "dice_rate_differences": (
            rate_differences
        ),
    }

    save_json(
        content=report,
        output_path=arguments.output_path,
    )

    dice_result = metric_differences["dice"]
    sensitivity_result = (
        metric_differences["sensitivity"]
    )

    unreliable_rate_result = (
        rate_differences["dice_below_0_70"]
    )

    excellent_rate_result = (
        rate_differences[
            "dice_at_least_0_90"
        ]
    )

    print(
        "Comparison direction: "
        "benign minus malignant"
    )

    print(
        "Dice difference: "
        f"{dice_result[
            'difference_benign_minus_malignant'
        ]:.4f} "
        f"(95% CI "
        f"{dice_result['ci_lower']:.4f}"
        f"–{dice_result['ci_upper']:.4f})"
    )

    print(
        "Sensitivity difference: "
        f"{sensitivity_result[
            'difference_benign_minus_malignant'
        ]:.4f} "
        f"(95% CI "
        f"{sensitivity_result['ci_lower']:.4f}"
        f"–{sensitivity_result['ci_upper']:.4f})"
    )

    print(
        "Dice < 0.70 rate difference: "
        f"{unreliable_rate_result[
            'difference_benign_minus_malignant'
        ]:.4f} "
        f"(95% CI "
        f"{unreliable_rate_result['ci_lower']:.4f}"
        f"–{unreliable_rate_result['ci_upper']:.4f})"
    )

    print(
        "Dice >= 0.90 rate difference: "
        f"{excellent_rate_result[
            'difference_benign_minus_malignant'
        ]:.4f} "
        f"(95% CI "
        f"{excellent_rate_result['ci_lower']:.4f}"
        f"–{excellent_rate_result['ci_upper']:.4f})"
    )

    print(
        f"Report saved to: "
        f"{arguments.output_path}"
    )


if __name__ == "__main__":
    main()
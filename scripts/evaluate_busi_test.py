"""Evaluate the final U-Net checkpoint on the untouched BUSI test split."""

from __future__ import annotations


import argparse
import csv
import json
from pathlib import Path
import math

import torch

from src.data import create_busi_dataloaders
from src.metrics import (
    binary_segmentation_statistics_per_image_from_logits,
)
from src.model import UNet
from src.reproducibility import seed_everything


REPORT_METRICS = (
    "dice",
    "iou",
    "precision",
    "sensitivity",
    "specificity",
)


def parse_arguments() -> argparse.Namespace:
    """Read final test-evaluation settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained U-Net on the untouched "
            "BUSI test split."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/BUSI/BUSI"),
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/splits/busi_seed42.csv"),
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "artifacts/busi_baseline_128_seed42/"
            "best_unet.pt"
        ),
    )

    parser.add_argument(
        "--threshold-selection-path",
        type=Path,
        default=Path(
            "artifacts/busi_baseline_128_seed42/"
            "threshold_selection.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/busi_baseline_128_seed42/"
            "test_evaluation"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dropout-probability",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
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
    """Reject invalid paths and numerical settings."""

    if not arguments.data_root.is_dir():
        raise FileNotFoundError(
            f"BUSI data directory was not found: "
            f"{arguments.data_root}"
        )

    if not arguments.manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest was not found: "
            f"{arguments.manifest_path}"
        )

    if not arguments.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: "
            f"{arguments.checkpoint_path}"
        )

    if not arguments.threshold_selection_path.is_file():
        raise FileNotFoundError(
            f"Threshold-selection file was not found: "
            f"{arguments.threshold_selection_path}"
        )

    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    if arguments.image_size <= 0:
        raise ValueError(
            "Image size must be greater than zero."
        )

    if arguments.num_workers < 0:
        raise ValueError(
            "Number of workers cannot be negative."
        )

    if arguments.base_channels <= 0:
        raise ValueError(
            "Base channels must be greater than zero."
        )

    if not 0.0 <= arguments.dropout_probability < 1.0:
        raise ValueError(
            "Dropout probability must be in [0, 1)."
        )

    if arguments.bootstrap_iterations <= 0:
        raise ValueError(
            "Bootstrap iterations must be greater than zero."
        )

    if not 0.0 < arguments.confidence_level < 1.0:
        raise ValueError(
            "Confidence level must be between zero and one."
        )


def load_locked_threshold(
    threshold_selection_path: Path,
) -> tuple[float, dict[str, object]]:
    """Load a threshold selected exclusively on validation data."""

    with threshold_selection_path.open(
        mode="r",
        encoding="utf-8",
    ) as threshold_file:
        threshold_results = json.load(
            threshold_file
        )

    if (
        threshold_results.get("selection_split")
        != "validation"
    ):
        raise ValueError(
            "The segmentation threshold must be selected "
            "using the validation split."
        )

    threshold = float(
        threshold_results["selected_threshold"]
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Selected threshold must be between zero and one."
        )

    return threshold, threshold_results


def validate_threshold_checkpoint_consistency(
    threshold_results: dict[str, object],
    checkpoint: dict[str, object],
) -> None:
    """Ensure that the threshold belongs to the loaded checkpoint."""

    threshold_checkpoint_epoch = int(
        threshold_results["checkpoint_epoch"]
    )

    loaded_checkpoint_epoch = int(
        checkpoint["epoch"]
    )

    if threshold_checkpoint_epoch != loaded_checkpoint_epoch:
        raise ValueError(
            "Threshold-selection checkpoint epoch does not "
            "match the loaded model checkpoint. "
            f"Threshold epoch: {threshold_checkpoint_epoch}; "
            f"loaded checkpoint epoch: {loaded_checkpoint_epoch}."
        )

    threshold_validation_dice = float(
        threshold_results["checkpoint_validation_dice"]
    )

    loaded_validation_dice = float(
        checkpoint["validation_dice"]
    )

    if not math.isclose(
        threshold_validation_dice,
        loaded_validation_dice,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Threshold-selection validation Dice does not "
            "match the loaded model checkpoint. "
            f"Threshold file Dice: {threshold_validation_dice}; "
            f"checkpoint Dice: {loaded_validation_dice}."
        )

def collect_test_outputs(
    model: UNet,
    test_loader,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
]:
    """Run the test split once and retain outputs and metadata."""

    all_logits = []
    all_targets = []
    class_names: list[str] = []
    filenames: list[str] = []

    model.eval()

    with torch.inference_mode():
        for batch in test_loader:
            images = batch["image"].to(
                device=device,
                non_blocking=True,
            )

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_targets.append(
                batch["mask"].cpu()
            )

            class_names.extend(
                batch["class_name"]
            )

            filenames.extend(
                batch["filename"]
            )

    if not all_logits:
        raise RuntimeError(
            "Test DataLoader did not provide any samples."
        )

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_targets, dim=0),
        class_names,
        filenames,
    )


def summarize_values(
    values: torch.Tensor,
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, int | float]:
    """Summarize one image-level metric with a bootstrap CI."""

    values = values.detach().cpu().to(
        dtype=torch.float64
    )

    if values.ndim != 1:
        raise ValueError(
            "Metric values must be one-dimensional."
        )

    sample_count = values.numel()

    if sample_count == 0:
        raise ValueError(
            "Metric summary requires at least one sample."
        )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    indices = torch.randint(
        low=0,
        high=sample_count,
        size=(
            bootstrap_iterations,
            sample_count,
        ),
        generator=generator,
    )

    bootstrap_means = values[
        indices
    ].mean(dim=1)

    alpha = (
        1.0 - confidence_level
    ) / 2.0

    lower = torch.quantile(
        bootstrap_means,
        alpha,
    )

    upper = torch.quantile(
        bootstrap_means,
        1.0 - alpha,
    )

    return {
        "sample_count": sample_count,
        "mean": float(values.mean()),
        "median": float(
            torch.quantile(values, 0.5)
        ),
        "standard_deviation": float(
            values.std(
                unbiased=sample_count > 1,
            )
        ),
        "confidence_level": confidence_level,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def summarize_metric_collection(
    statistics: dict[str, torch.Tensor],
    selector: torch.Tensor,
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, dict[str, int | float]]:
    """Summarize all reported metrics for selected images."""

    summary = {}

    for metric_index, metric_name in enumerate(
        REPORT_METRICS
    ):
        summary[metric_name] = summarize_values(
            values=statistics[metric_name][selector],
            bootstrap_iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            seed=seed + metric_index,
        )

    return summary


def compute_micro_metrics(
    statistics: dict[str, torch.Tensor],
    selector: torch.Tensor,
    epsilon: float = 1e-7,
) -> dict[str, float]:
    """Compute pooled pixel-level metrics."""

    true_positive = float(
        statistics["true_positive"][
            selector
        ].sum()
    )

    false_positive = float(
        statistics["false_positive"][
            selector
        ].sum()
    )

    false_negative = float(
        statistics["false_negative"][
            selector
        ].sum()
    )

    true_negative = float(
        statistics["true_negative"][
            selector
        ].sum()
    )

    return {
        "dice": (
            2.0 * true_positive + epsilon
        ) / (
            2.0 * true_positive
            + false_positive
            + false_negative
            + epsilon
        ),
        "iou": (
            true_positive + epsilon
        ) / (
            true_positive
            + false_positive
            + false_negative
            + epsilon
        ),
        "precision": true_positive / (
            true_positive
            + false_positive
            + epsilon
        ),
        "sensitivity": true_positive / (
            true_positive
            + false_negative
            + epsilon
        ),
        "specificity": true_negative / (
            true_negative
            + false_positive
            + epsilon
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def build_per_image_rows(
    filenames: list[str],
    class_names: list[str],
    statistics: dict[str, torch.Tensor],
) -> list[dict[str, str | float]]:
    """Create one serializable result row per image."""

    rows = []

    for index, filename in enumerate(
        filenames
    ):
        target_fraction = float(
            statistics[
                "target_positive_fraction"
            ][index]
        )

        predicted_fraction = float(
            statistics[
                "predicted_positive_fraction"
            ][index]
        )

        area_ratio = (
            predicted_fraction / target_fraction
            if target_fraction > 0.0
            else 0.0
        )

        rows.append(
            {
                "filename": filename,
                "class_name": class_names[index],
                "dice": float(
                    statistics["dice"][index]
                ),
                "iou": float(
                    statistics["iou"][index]
                ),
                "precision": float(
                    statistics["precision"][index]
                ),
                "sensitivity": float(
                    statistics["sensitivity"][index]
                ),
                "specificity": float(
                    statistics["specificity"][index]
                ),
                "true_positive": float(
                    statistics["true_positive"][index]
                ),
                "false_positive": float(
                    statistics["false_positive"][index]
                ),
                "false_negative": float(
                    statistics["false_negative"][index]
                ),
                "true_negative": float(
                    statistics["true_negative"][index]
                ),
                "predicted_positive_fraction": (
                    predicted_fraction
                ),
                "target_positive_fraction": (
                    target_fraction
                ),
                "predicted_to_target_area_ratio": (
                    area_ratio
                ),
            }
        )

    return rows


def save_per_image_results(
    rows: list[dict[str, str | float]],
    output_path: Path,
) -> None:
    """Save image-level results as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_json(
    content: dict[str, object],
    output_path: Path,
) -> None:
    """Save an evaluation report as JSON."""

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
    """Evaluate the final model once on the BUSI test split."""

    arguments = parse_arguments()
    validate_arguments(arguments)
    seed_everything(arguments.seed)

    (
        selected_threshold,
        threshold_results,
    ) = load_locked_threshold(
        arguments.threshold_selection_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_loaders = create_busi_dataloaders(
        root=arguments.data_root,
        manifest_path=arguments.manifest_path,
        batch_size=arguments.batch_size,
        image_size=arguments.image_size,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=arguments.base_channels,
        dropout_probability=(
            arguments.dropout_probability
        ),
    ).to(device)

    checkpoint = torch.load(
        arguments.checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    validate_threshold_checkpoint_consistency(
        threshold_results=threshold_results,
        checkpoint=checkpoint,
    )

    (
        logits,
        targets,
        class_names,
        filenames,
    ) = collect_test_outputs(
        model=model,
        test_loader=data_loaders["test"],
        device=device,
    )

    statistics = (
        binary_segmentation_statistics_per_image_from_logits(
            logits=logits,
            targets=targets,
            threshold=selected_threshold,
        )
    )

    all_selector = torch.ones(
        len(filenames),
        dtype=torch.bool,
    )

    class_selectors = {
        class_name: torch.tensor(
            [
                sample_class == class_name
                for sample_class in class_names
            ],
            dtype=torch.bool,
        )
        for class_name in (
            "benign",
            "malignant",
        )
    }

    overall_summary = summarize_metric_collection(
        statistics=statistics,
        selector=all_selector,
        bootstrap_iterations=(
            arguments.bootstrap_iterations
        ),
        confidence_level=(
            arguments.confidence_level
        ),
        seed=arguments.seed,
    )

    class_summaries = {}

    for class_index, (
        class_name,
        selector,
    ) in enumerate(
        class_selectors.items()
    ):
        class_summaries[class_name] = {
            "image_level": (
                summarize_metric_collection(
                    statistics=statistics,
                    selector=selector,
                    bootstrap_iterations=(
                        arguments.bootstrap_iterations
                    ),
                    confidence_level=(
                        arguments.confidence_level
                    ),
                    seed=(
                        arguments.seed
                        + 100
                        + class_index * 10
                    ),
                )
            ),
            "pooled_pixel_level": (
                compute_micro_metrics(
                    statistics=statistics,
                    selector=selector,
                )
            ),
        }

    per_image_rows = build_per_image_rows(
        filenames=filenames,
        class_names=class_names,
        statistics=statistics,
    )

    sorted_by_dice = sorted(
        per_image_rows,
        key=lambda row: float(row["dice"]),
    )

    report = {
        "evaluation_split": "test",
        "test_was_not_used_for_model_selection": True,
        "test_sample_count": len(filenames),
        "checkpoint_epoch": checkpoint["epoch"],
        "threshold": selected_threshold,
        "threshold_source": "validation",
        "validation_threshold_selection": {
            "validation_sample_count": (
                threshold_results[
                    "validation_sample_count"
                ]
            ),
            "selected_validation_metrics": (
                threshold_results[
                    "selected_metrics"
                ]
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
        },
        "overall": {
            "image_level": overall_summary,
            "pooled_pixel_level": (
                compute_micro_metrics(
                    statistics=statistics,
                    selector=all_selector,
                )
            ),
        },
        "class_specific": class_summaries,
        "lowest_dice_examples": (
            sorted_by_dice[:10]
        ),
        "highest_dice_examples": (
            list(
                reversed(
                    sorted_by_dice[-10:]
                )
            )
        ),
    }

    report_path = (
        arguments.output_dir
        / "test_results.json"
    )

    per_image_path = (
        arguments.output_dir
        / "per_image_results.csv"
    )

    save_json(
        content=report,
        output_path=report_path,
    )

    save_per_image_results(
        rows=per_image_rows,
        output_path=per_image_path,
    )

    print(f"Device: {device}")
    print(
        f"Test samples: {len(filenames)}"
    )
    print(
        f"Locked threshold: "
        f"{selected_threshold:.2f}"
    )
    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )
    print()

    for metric_name in REPORT_METRICS:
        metric_summary = overall_summary[
            metric_name
        ]

        print(
            f"{metric_name.capitalize()}: "
            f"{metric_summary['mean']:.4f} "
            f"(95% CI "
            f"{metric_summary['ci_lower']:.4f}"
            f"–{metric_summary['ci_upper']:.4f})"
        )

    print()
    print(f"Report saved to: {report_path}")
    print(
        f"Per-image results saved to: "
        f"{per_image_path}"
    )


if __name__ == "__main__":
    main()
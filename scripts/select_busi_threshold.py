"""Select a segmentation threshold using only the BUSI validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import create_busi_dataloaders
from src.metrics import (
    binary_segmentation_statistics_per_image_from_logits,
)
from src.model import UNet
from src.reproducibility import seed_everything


def parse_arguments() -> argparse.Namespace:
    """Read threshold-selection settings from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Select a binary segmentation threshold using "
            "only the BUSI validation split."
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
        "--output-path",
        type=Path,
        default=Path(
            "artifacts/busi_baseline_128_seed42/"
            "threshold_selection.json"
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
        "--seed",
        type=int,
        default=42,
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

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Validate paths and numerical settings."""

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


def collect_validation_outputs(
    model: UNet,
    validation_loader,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
]:
    """Run validation once and collect logits and metadata."""

    all_logits = []
    all_targets = []
    class_names: list[str] = []
    filenames: list[str] = []

    model.eval()

    with torch.inference_mode():
        for batch in validation_loader:
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
            "Validation DataLoader did not provide any samples."
        )

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_targets, dim=0),
        class_names,
        filenames,
    )


def evaluate_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Compute mean validation metrics for one threshold."""

    statistics = (
        binary_segmentation_statistics_per_image_from_logits(
            logits=logits,
            targets=targets,
            threshold=threshold,
        )
    )

    return {
        "threshold": threshold,
        "dice": float(
            statistics["dice"].mean()
        ),
        "iou": float(
            statistics["iou"].mean()
        ),
        "precision": float(
            statistics["precision"].mean()
        ),
        "sensitivity": float(
            statistics["sensitivity"].mean()
        ),
        "specificity": float(
            statistics["specificity"].mean()
        ),
    }


def select_best_result(
    results: list[dict[str, float]],
) -> dict[str, float]:
    """Select the highest-Dice threshold deterministically."""

    return max(
        results,
        key=lambda result: (
            result["dice"],
            result["iou"],
            -abs(result["threshold"] - 0.5),
        ),
    )


def summarize_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
    class_name: str,
    threshold: float,
) -> dict[str, int | float]:
    """Summarize validation performance for one diagnostic class."""

    selector = torch.tensor(
        [
            sample_class == class_name
            for sample_class in class_names
        ],
        dtype=torch.bool,
    )

    statistics = (
        binary_segmentation_statistics_per_image_from_logits(
            logits=logits[selector],
            targets=targets[selector],
            threshold=threshold,
        )
    )

    return {
        "sample_count": int(
            selector.sum().item()
        ),
        "mean_dice": float(
            statistics["dice"].mean()
        ),
        "mean_iou": float(
            statistics["iou"].mean()
        ),
        "mean_precision": float(
            statistics["precision"].mean()
        ),
        "mean_sensitivity": float(
            statistics["sensitivity"].mean()
        ),
        "mean_specificity": float(
            statistics["specificity"].mean()
        ),
    }


def save_results(
    results: dict[str, object],
    output_path: Path,
) -> None:
    """Save threshold-selection results as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            results,
            output_file,
            indent=2,
        )


def main() -> None:
    """Select and save the validation threshold."""

    arguments = parse_arguments()
    validate_arguments(arguments)
    seed_everything(arguments.seed)

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

    (
        logits,
        targets,
        class_names,
        filenames,
    ) = collect_validation_outputs(
        model=model,
        validation_loader=data_loaders["validation"],
        device=device,
    )

    coarse_thresholds = [
        value / 100
        for value in range(10, 91, 5)
    ]

    coarse_results = [
        evaluate_threshold(
            logits=logits,
            targets=targets,
            threshold=threshold,
        )
        for threshold in coarse_thresholds
    ]

    best_coarse = select_best_result(
        coarse_results
    )

    fine_start = max(
        0.01,
        best_coarse["threshold"] - 0.05,
    )

    fine_end = min(
        0.99,
        best_coarse["threshold"] + 0.05,
    )

    fine_thresholds = []

    threshold = fine_start

    while threshold <= fine_end + 1e-9:
        fine_thresholds.append(
            round(threshold, 2)
        )
        threshold += 0.01

    fine_results = [
        evaluate_threshold(
            logits=logits,
            targets=targets,
            threshold=threshold,
        )
        for threshold in fine_thresholds
    ]

    selected_result = select_best_result(
        fine_results
    )

    selected_threshold = selected_result[
        "threshold"
    ]

    output = {
        "selection_split": "validation",
        "validation_sample_count": len(filenames),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_dice": (
            checkpoint["validation_dice"]
        ),
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_result,
        "class_specific_metrics": {
            "benign": summarize_class(
                logits=logits,
                targets=targets,
                class_names=class_names,
                class_name="benign",
                threshold=selected_threshold,
            ),
            "malignant": summarize_class(
                logits=logits,
                targets=targets,
                class_names=class_names,
                class_name="malignant",
                threshold=selected_threshold,
            ),
        },
        "coarse_search": coarse_results,
        "fine_search": fine_results,
    }

    save_results(
        results=output,
        output_path=arguments.output_path,
    )

    print(f"Device: {device}")
    print(
        f"Validation samples: "
        f"{len(filenames)}"
    )
    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )
    print(
        f"Validation Dice: "
        f"{selected_result['dice']:.4f}"
    )
    print(
        f"Validation IoU: "
        f"{selected_result['iou']:.4f}"
    )
    print(
        f"Results saved to: "
        f"{arguments.output_path}"
    )


if __name__ == "__main__":
    main()
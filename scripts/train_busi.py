"""Train the compact U-Net model on the BUSI dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW

from src.data import create_busi_dataloaders
from src.losses import BCEDiceLoss
from src.model import UNet
from src.reproducibility import seed_everything
from src.training import fit_model


def parse_arguments() -> argparse.Namespace:
    """Read training settings from command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a compact U-Net model on reproducible "
            "BUSI dataset splits."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/BUSI/BUSI"),
        help="Directory containing BUSI images and labels.",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/splits/busi_seed42.csv"),
        help="CSV file containing reproducible dataset splits.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/busi_baseline"),
        help="Directory used for checkpoints and training history.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of complete training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images in each batch.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square image size used by the model.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial AdamW learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight-decay coefficient.",
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
        help="Number of feature channels in the first U-Net level.",
    )

    parser.add_argument(
        "--dropout-probability",
        type=float,
        default=0.2,
        help="Dropout probability used in the U-Net bottleneck.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for Dice and IoU.",
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Reject invalid experiment settings before training starts."""

    if arguments.epochs <= 0:
        raise ValueError(
            "Epoch count must be greater than zero."
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

    if arguments.learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if arguments.weight_decay < 0:
        raise ValueError(
            "Weight decay cannot be negative."
        )

    if arguments.base_channels <= 0:
        raise ValueError(
            "Base channels must be greater than zero."
        )

    if not 0.0 <= arguments.dropout_probability < 1.0:
        raise ValueError(
            "Dropout probability must be in [0, 1)."
        )

    if not 0.0 < arguments.threshold < 1.0:
        raise ValueError(
            "Threshold must be between zero and one."
        )

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


def save_training_history(
    history: list[dict[str, int | float]],
    history_path: Path,
) -> None:
    """Save epoch-level training results as readable JSON."""

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with history_path.open(
        mode="w",
        encoding="utf-8",
    ) as history_file:
        json.dump(
            history,
            history_file,
            indent=2,
        )


def main() -> None:
    """Run one complete BUSI training experiment."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    seed_everything(arguments.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Data root: {arguments.data_root}")
    print(f"Manifest: {arguments.manifest_path}")
    print(f"Output directory: {arguments.output_dir}")

    data_loaders = create_busi_dataloaders(
        root=arguments.data_root,
        manifest_path=arguments.manifest_path,
        batch_size=arguments.batch_size,
        image_size=arguments.image_size,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )

    print(
        "Dataset sizes: "
        f"train={len(data_loaders['train'].dataset)}, "
        f"validation="
        f"{len(data_loaders['validation'].dataset)}, "
        f"test={len(data_loaders['test'].dataset)}"
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=arguments.base_channels,
        dropout_probability=(
            arguments.dropout_probability
        ),
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        smooth=1.0,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        arguments.output_dir
        / "best_unet.pt"
    )

    history_path = (
        arguments.output_dir
        / "history.json"
    )

    history = fit_model(
        model=model,
        train_loader=data_loaders["train"],
        validation_loader=data_loaders["validation"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=arguments.epochs,
        checkpoint_path=checkpoint_path,
        threshold=arguments.threshold,
    )

    save_training_history(
        history=history,
        history_path=history_path,
    )

    best_epoch_record = max(
        history,
        key=lambda record: record["validation_dice"],
    )

    print()
    print("Training completed.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(
        f"Best epoch: "
        f"{best_epoch_record['epoch']}"
    )
    print(
        f"Best validation Dice: "
        f"{best_epoch_record['validation_dice']:.4f}"
    )
    print(
        f"Best validation IoU: "
        f"{best_epoch_record['validation_iou']:.4f}"
    )


if __name__ == "__main__":
    main()
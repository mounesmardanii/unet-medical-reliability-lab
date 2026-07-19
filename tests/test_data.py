"""Tests for BUSI dataset and DataLoader utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from src.data import (
    BUSIDataset,
    create_busi_dataloaders,
    load_busi_split_samples,
)


def create_fake_busi_data(
    temporary_directory: Path,
) -> tuple[Path, Path]:
    """Create a small BUSI-like dataset for isolated tests."""

    root = temporary_directory / "BUSI"
    images_directory = root / "images"
    labels_directory = root / "labels"

    images_directory.mkdir(parents=True)
    labels_directory.mkdir(parents=True)

    manifest_path = temporary_directory / "splits.csv"

    rows = [
        {
            "filename": "benign_1.png",
            "class_name": "benign",
            "split": "train",
        },
        {
            "filename": "malignant_1.png",
            "class_name": "malignant",
            "split": "train",
        },
        {
            "filename": "benign_2.png",
            "class_name": "benign",
            "split": "validation",
        },
        {
            "filename": "malignant_2.png",
            "class_name": "malignant",
            "split": "validation",
        },
        {
            "filename": "benign_3.png",
            "class_name": "benign",
            "split": "test",
        },
        {
            "filename": "malignant_3.png",
            "class_name": "malignant",
            "split": "test",
        },
    ]

    for index, row in enumerate(rows):
        image = Image.new(
            mode="L",
            size=(20, 16),
            color=60 + index * 20,
        )

        mask = Image.new(
            mode="L",
            size=(20, 16),
            color=0,
        )

        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            xy=(5, 4, 14, 11),
            fill=1,
        )

        image.save(
            images_directory / row["filename"]
        )

        mask.save(
            labels_directory / row["filename"]
        )

    with manifest_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "filename",
                "class_name",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return root, manifest_path


def test_load_busi_split_samples(
    tmp_path: Path,
) -> None:
    """Manifest loading should return only the requested split."""

    root, manifest_path = create_fake_busi_data(
        tmp_path
    )

    train_samples = load_busi_split_samples(
        root=root,
        manifest_path=manifest_path,
        split="train",
    )

    assert len(train_samples) == 2

    assert {
        sample.image_path.name
        for sample in train_samples
    } == {
        "benign_1.png",
        "malignant_1.png",
    }


def test_busi_dataset_returns_binary_tensors(
    tmp_path: Path,
) -> None:
    """BUSIDataset should return correctly shaped binary masks."""

    root, manifest_path = create_fake_busi_data(
        tmp_path
    )

    samples = load_busi_split_samples(
        root=root,
        manifest_path=manifest_path,
        split="validation",
    )

    dataset = BUSIDataset(
        samples=samples,
        image_size=32,
        augment=False,
    )

    item = dataset[0]

    assert item["image"].shape == (1, 32, 32)
    assert item["mask"].shape == (1, 32, 32)

    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.float32

    assert set(
        torch.unique(item["mask"]).tolist()
    ).issubset({0.0, 1.0})


def test_create_busi_dataloaders(
    tmp_path: Path,
) -> None:
    """DataLoaders should use correct split and augmentation settings."""

    root, manifest_path = create_fake_busi_data(
        tmp_path
    )

    loaders = create_busi_dataloaders(
        root=root,
        manifest_path=manifest_path,
        batch_size=2,
        image_size=32,
        num_workers=0,
        seed=42,
    )

    assert len(loaders["train"].dataset) == 2
    assert len(loaders["validation"].dataset) == 2
    assert len(loaders["test"].dataset) == 2

    assert loaders["train"].dataset.augment is True
    assert loaders["validation"].dataset.augment is False
    assert loaders["test"].dataset.augment is False

    train_batch = next(
        iter(loaders["train"])
    )

    assert train_batch["image"].shape == (
        2,
        1,
        32,
        32,
    )

    assert train_batch["mask"].shape == (
        2,
        1,
        32,
        32,
    )

    assert set(
        torch.unique(
            train_batch["mask"]
        ).tolist()
    ).issubset({0.0, 1.0})


def test_invalid_split_is_rejected(
    tmp_path: Path,
) -> None:
    """Unknown split names should raise a clear error."""

    root, manifest_path = create_fake_busi_data(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="Unknown split",
    ):
        load_busi_split_samples(
            root=root,
            manifest_path=manifest_path,
            split="development",
        )
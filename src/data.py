"""Dataset utilities for the BUSI breast ultrasound dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageStat
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}


@dataclass(frozen=True)
class BUSISample:
    """Paths and metadata for one BUSI image-mask pair."""

    image_path: Path
    mask_path: Path
    class_name: str
    class_index: int


def discover_busi_samples(
    root: str | Path,
) -> list[BUSISample]:
    """Find and validate all BUSI image-mask pairs."""

    root = Path(root)

    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}"
        )

    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Labels directory not found: {labels_dir}"
        )

    samples: list[BUSISample] = []

    for image_path in sorted(images_dir.glob("*.png")):
        class_name = image_path.stem.split(
            "_",
            maxsplit=1,
        )[0]

        if class_name not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown class in filename: {image_path.name}"
            )

        mask_path = labels_dir / image_path.name

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Mask not found for image: {image_path.name}"
            )

        samples.append(
            BUSISample(
                image_path=image_path,
                mask_path=mask_path,
                class_name=class_name,
                class_index=CLASS_TO_INDEX[class_name],
            )
        )

    if not samples:
        raise RuntimeError(
            f"No PNG images found in: {images_dir}"
        )

    return samples


def load_busi_split_samples(
    root: str | Path,
    manifest_path: str | Path,
    split: str,
) -> list[BUSISample]:
    """Load one BUSI split from a saved CSV manifest."""

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    if split not in valid_splits:
        raise ValueError(
            f"Unknown split: {split}. "
            f"Expected one of: {sorted(valid_splits)}"
        )

    root = Path(root)
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}"
        )

    images_dir = root / "images"
    labels_dir = root / "labels"

    required_columns = {
        "filename",
        "class_name",
        "split",
    }

    samples: list[BUSISample] = []

    with manifest_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            if row["split"] != split:
                continue

            filename = row["filename"]
            class_name = row["class_name"]

            if class_name not in CLASS_TO_INDEX:
                raise ValueError(
                    "Unknown class in manifest row: "
                    f"{class_name}"
                )

            image_path = images_dir / filename
            mask_path = labels_dir / filename

            if not image_path.is_file():
                raise FileNotFoundError(
                    "Image listed in manifest was not found: "
                    f"{image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    "Mask listed in manifest was not found: "
                    f"{mask_path}"
                )

            samples.append(
                BUSISample(
                    image_path=image_path,
                    mask_path=mask_path,
                    class_name=class_name,
                    class_index=CLASS_TO_INDEX[class_name],
                )
            )

    if not samples:
        raise RuntimeError(
            f"No samples found for split: {split}"
        )

    return samples


class BUSIDataset(Dataset):
    """PyTorch dataset for BUSI ultrasound segmentation."""

    def __init__(
        self,
        samples: list[BUSISample],
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        if not samples:
            raise ValueError(
                "The sample list must not be empty."
            )

        if image_size <= 0:
            raise ValueError(
                "Image size must be greater than zero."
            )

        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        """Return the number of dataset samples."""

        return len(self.samples)

    def _apply_joint_augmentation(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """Apply identical random spatial transforms to image and mask."""

        # Flip both the image and mask horizontally
        # with a probability of 50%.
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Rotate both the image and mask
        # with a probability of 50%.
        if torch.rand(1).item() < 0.5:
            angle = float(
                torch.empty(1)
                .uniform_(-10.0, 10.0)
                .item()
            )

            # Use the median image intensity as the fill value
            # to avoid creating artificial black corners.
            fill_value = int(
                ImageStat.Stat(image).median[0]
            )

            image = TF.rotate(
                image,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=fill_value,
            )

            # Nearest-neighbor interpolation preserves
            # the discrete values of the binary mask.
            mask = TF.rotate(
                mask,
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        return image, mask

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:
        """Load and preprocess one BUSI image-mask pair."""

        sample = self.samples[index]

        with (
            Image.open(sample.image_path) as image_file,
            Image.open(sample.mask_path) as mask_file,
        ):
            image = image_file.convert("L")
            mask = mask_file.convert("L")

            if self.augment:
                image, mask = self._apply_joint_augmentation(
                    image=image,
                    mask=mask,
                )

            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )

            image_tensor = TF.to_tensor(image)

            # Convert every positive mask pixel to 1
            # and keep background pixels equal to 0.
            mask_tensor = (
                TF.pil_to_tensor(mask) > 0
            ).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "class_index": torch.tensor(
                sample.class_index,
                dtype=torch.long,
            ),
            "class_name": sample.class_name,
            "filename": sample.image_path.name,
        }


def create_busi_dataloaders(
    root: str | Path,
    manifest_path: str | Path,
    batch_size: int = 8,
    image_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create reproducible BUSI train, validation, and test loaders."""

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    if image_size <= 0:
        raise ValueError(
            "Image size must be greater than zero."
        )

    if num_workers < 0:
        raise ValueError(
            "Number of workers must not be negative."
        )

    train_samples = load_busi_split_samples(
        root=root,
        manifest_path=manifest_path,
        split="train",
    )

    validation_samples = load_busi_split_samples(
        root=root,
        manifest_path=manifest_path,
        split="validation",
    )

    test_samples = load_busi_split_samples(
        root=root,
        manifest_path=manifest_path,
        split="test",
    )

    # Augmentation is enabled only for the training set.
    train_dataset = BUSIDataset(
        samples=train_samples,
        image_size=image_size,
        augment=True,
    )

    # Validation data must remain unchanged so that
    # different epochs can be compared fairly.
    validation_dataset = BUSIDataset(
        samples=validation_samples,
        image_size=image_size,
        augment=False,
    )

    # Test data must also remain unchanged because it is
    # reserved for the final evaluation of the model.
    test_dataset = BUSIDataset(
        samples=test_samples,
        image_size=image_size,
        augment=False,
    )

    # This generator makes the initial training shuffle
    # reproducible when the same seed is used.
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    common_loader_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **common_loader_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_arguments,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_arguments,
    )

    return {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }
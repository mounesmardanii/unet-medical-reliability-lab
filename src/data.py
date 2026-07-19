"""Dataset utilities for the BUSI breast ultrasound dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageStat
from torch.utils.data import Dataset
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

        # Apply horizontal flipping with 50% probability.
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Apply a small rotation with 50% probability.
        if torch.rand(1).item() < 0.5:
            angle = float(
                torch.empty(1)
                .uniform_(-10.0, 10.0)
                .item()
            )

            # Use the image median as the rotation fill value
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

            # Keep the segmentation mask strictly binary.
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
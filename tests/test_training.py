"""Tests for segmentation training and evaluation utilities."""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.losses import BCEDiceLoss
from src.model import UNet
from src.reproducibility import seed_everything
from src.training import (
    evaluate_one_epoch,
    train_one_epoch,
)


def create_fake_data_loader(
    sample_count: int = 4,
    batch_size: int = 2,
) -> DataLoader:
    """Create a small segmentation DataLoader for isolated tests."""

    samples = []

    for _ in range(sample_count):
        image = torch.rand(
            1,
            32,
            32,
        )

        mask = torch.zeros(
            1,
            32,
            32,
        )

        mask[
            :,
            8:24,
            8:24,
        ] = 1.0

        samples.append(
            {
                "image": image,
                "mask": mask,
            }
        )

    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
    )


def create_small_unet() -> UNet:
    """Create a compact U-Net suitable for fast unit tests."""

    return UNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        dropout_probability=0.10,
    )


def test_train_one_epoch_updates_model_parameters() -> None:
    """Training should return finite loss and update model weights."""

    seed_everything(42)

    device = torch.device("cpu")

    data_loader = create_fake_data_loader()
    model = create_small_unet().to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    first_parameter_before = next(
        model.parameters()
    ).detach().clone()

    result = train_one_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    first_parameter_after = next(
        model.parameters()
    ).detach().clone()

    assert set(result) == {"loss"}

    assert torch.isfinite(
        torch.tensor(result["loss"])
    )

    assert result["loss"] > 0
    assert model.training is True

    assert not torch.equal(
        first_parameter_before,
        first_parameter_after,
    )


def test_evaluate_one_epoch_preserves_parameters() -> None:
    """Evaluation should return metrics without updating model weights."""

    seed_everything(42)

    device = torch.device("cpu")

    data_loader = create_fake_data_loader()
    model = create_small_unet().to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    first_parameter_before = next(
        model.parameters()
    ).detach().clone()

    result = evaluate_one_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        threshold=0.5,
    )

    first_parameter_after = next(
        model.parameters()
    ).detach().clone()

    assert set(result) == {
        "loss",
        "dice",
        "iou",
    }

    assert torch.isfinite(
        torch.tensor(result["loss"])
    )

    assert 0.0 <= result["dice"] <= 1.0
    assert 0.0 <= result["iou"] <= 1.0

    assert model.training is False

    assert torch.equal(
        first_parameter_before,
        first_parameter_after,
    )

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_train_one_epoch_rejects_empty_loader() -> None:
    """Training should reject a DataLoader without samples."""

    device = torch.device("cpu")

    empty_loader = DataLoader(
        [],
        batch_size=2,
    )

    model = create_small_unet().to(device)
    criterion = BCEDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(
        RuntimeError,
        match="did not provide any samples",
    ):
        train_one_epoch(
            model=model,
            data_loader=empty_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )


def test_evaluate_one_epoch_rejects_empty_loader() -> None:
    """Evaluation should reject a DataLoader without samples."""

    device = torch.device("cpu")

    empty_loader = DataLoader(
        [],
        batch_size=2,
    )

    model = create_small_unet().to(device)
    criterion = BCEDiceLoss()

    with pytest.raises(
        RuntimeError,
        match="did not provide any samples",
    ):
        evaluate_one_epoch(
            model=model,
            data_loader=empty_loader,
            criterion=criterion,
            device=device,
            threshold=0.5,
        )
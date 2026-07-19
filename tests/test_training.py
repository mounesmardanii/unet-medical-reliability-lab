"""Tests for segmentation training and evaluation utilities."""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import src.training as training_module
from src.losses import BCEDiceLoss
from src.model import UNet
from src.reproducibility import seed_everything
from src.training import (
    evaluate_one_epoch,
    fit_model,
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


def test_fit_model_saves_best_checkpoint(
    tmp_path,
) -> None:
    """Multi-epoch training should save the best validation model."""

    seed_everything(42)

    device = torch.device("cpu")

    train_loader = create_fake_data_loader(
        sample_count=6,
        batch_size=2,
    )

    validation_loader = create_fake_data_loader(
        sample_count=4,
        batch_size=2,
    )

    model = create_small_unet().to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint_path = (
        tmp_path
        / "checkpoints"
        / "best_unet.pt"
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        checkpoint_path=checkpoint_path,
        threshold=0.5,
    )

    assert len(history) == 2

    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2

    expected_history_keys = {
        "epoch",
        "learning_rate",
        "train_loss",
        "validation_loss",
        "validation_dice",
        "validation_iou",
    }

    assert set(history[0]) == expected_history_keys
    assert set(history[1]) == expected_history_keys

    assert history[0]["learning_rate"] == pytest.approx(
        1e-3,
    )

    assert history[1]["learning_rate"] == pytest.approx(
        1e-3,
    )

    assert checkpoint_path.is_file()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    expected_checkpoint_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "learning_rate",
        "validation_loss",
        "validation_dice",
        "validation_iou",
        "threshold",
    }

    assert set(checkpoint) == expected_checkpoint_keys

    assert checkpoint["scheduler_state_dict"] is None

    assert checkpoint["learning_rate"] == pytest.approx(
        1e-3,
    )

    best_history_dice = max(
        record["validation_dice"]
        for record in history
    )

    assert checkpoint["epoch"] in {
        1,
        2,
    }

    assert checkpoint["threshold"] == pytest.approx(
        0.5,
    )

    assert checkpoint[
        "validation_dice"
    ] == pytest.approx(
        best_history_dice,
        abs=1e-12,
    )


def test_fit_model_rejects_invalid_epoch_count(
    tmp_path,
) -> None:
    """Training should reject nonpositive epoch counts."""

    device = torch.device("cpu")

    train_loader = create_fake_data_loader()
    validation_loader = create_fake_data_loader()

    model = create_small_unet().to(device)
    criterion = BCEDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        fit_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=0,
            checkpoint_path=(
                tmp_path
                / "best_unet.pt"
            ),
            threshold=0.5,
        )

def test_fit_model_reduces_learning_rate_on_plateau(
    tmp_path,
    monkeypatch,
) -> None:
    """Scheduler should reduce learning rate when Dice plateaus."""

    device = torch.device("cpu")

    train_loader = create_fake_data_loader()
    validation_loader = create_fake_data_loader()

    model = create_small_unet().to(device)
    criterion = BCEDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=1e-6,
    )

    def fake_train_one_epoch(
        **_: object,
    ) -> dict[str, float]:
        return {
            "loss": 0.8,
        }

    def fake_evaluate_one_epoch(
        **_: object,
    ) -> dict[str, float]:
        return {
            "loss": 0.7,
            "dice": 0.5,
            "iou": 0.35,
        }

    monkeypatch.setattr(
        training_module,
        "train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        training_module,
        "evaluate_one_epoch",
        fake_evaluate_one_epoch,
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        checkpoint_path=(
            tmp_path
            / "scheduler"
            / "best_unet.pt"
        ),
        threshold=0.5,
        scheduler=scheduler,
    )

    learning_rates = [
        record["learning_rate"]
        for record in history
    ]

    assert learning_rates == pytest.approx(
        [
            1e-3,
            1e-3,
            5e-4,
        ]
    )

    assert optimizer.param_groups[0][
        "lr"
    ] == pytest.approx(
        2.5e-4,
    )

def test_fit_model_stops_early_after_plateau(
    tmp_path,
    monkeypatch,
) -> None:
    """Training should stop after repeated insignificant epochs."""

    device = torch.device("cpu")

    train_loader = create_fake_data_loader()
    validation_loader = create_fake_data_loader()

    model = create_small_unet().to(device)
    criterion = BCEDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    def fake_train_one_epoch(
        **_: object,
    ) -> dict[str, float]:
        return {
            "loss": 0.8,
        }

    def fake_evaluate_one_epoch(
        **_: object,
    ) -> dict[str, float]:
        return {
            "loss": 0.7,
            "dice": 0.5,
            "iou": 0.35,
        }

    monkeypatch.setattr(
        training_module,
        "train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        training_module,
        "evaluate_one_epoch",
        fake_evaluate_one_epoch,
    )

    checkpoint_path = (
        tmp_path
        / "early_stopping"
        / "best_unet.pt"
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        checkpoint_path=checkpoint_path,
        threshold=0.5,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
    )

    assert len(history) == 3
    assert history[-1]["epoch"] == 3

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    assert checkpoint["epoch"] == 1

    assert checkpoint[
        "validation_dice"
    ] == pytest.approx(
        0.5,
    ) 


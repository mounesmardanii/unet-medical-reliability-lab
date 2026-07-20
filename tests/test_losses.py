"""Tests for segmentation loss functions."""

import pytest
import torch

from src.losses import BCEDiceLoss, DiceLoss


def test_dice_loss_rewards_correct_predictions() -> None:
    """Correct predictions should have lower Dice loss."""

    criterion = DiceLoss(smooth=1.0)

    targets = torch.ones(
        2,
        1,
        4,
        4,
    )

    correct_logits = torch.full_like(
        targets,
        20.0,
    )

    wrong_logits = torch.full_like(
        targets,
        -20.0,
    )

    correct_loss = criterion(
        correct_logits,
        targets,
    )

    wrong_loss = criterion(
        wrong_logits,
        targets,
    )

    assert torch.isfinite(correct_loss)
    assert torch.isfinite(wrong_loss)
    assert correct_loss.item() < 0.001
    assert wrong_loss.item() > 0.90
    assert correct_loss < wrong_loss


def test_bce_dice_loss_rewards_correct_predictions() -> None:
    """Correct predictions should have lower combined loss."""

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        smooth=1.0,
    )

    targets = torch.ones(
        2,
        1,
        4,
        4,
    )

    correct_logits = torch.full_like(
        targets,
        20.0,
    )

    wrong_logits = torch.full_like(
        targets,
        -20.0,
    )

    correct_loss = criterion(
        correct_logits,
        targets,
    )

    wrong_loss = criterion(
        wrong_logits,
        targets,
    )

    assert torch.isfinite(correct_loss)
    assert torch.isfinite(wrong_loss)
    assert correct_loss.item() < 0.001
    assert wrong_loss.item() > 1.0
    assert correct_loss < wrong_loss


def test_dice_loss_rejects_shape_mismatch() -> None:
    """DiceLoss should reject incompatible tensor shapes."""

    criterion = DiceLoss()

    logits = torch.zeros(
        2,
        1,
        8,
        8,
    )

    targets = torch.zeros(
        2,
        1,
        4,
        4,
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        criterion(
            logits,
            targets,
        )


def test_bce_dice_loss_produces_gradients() -> None:
    """Combined loss should support backpropagation."""

    criterion = BCEDiceLoss()

    logits = torch.randn(
        2,
        1,
        8,
        8,
        requires_grad=True,
    )

    targets = torch.randint(
        low=0,
        high=2,
        size=(2, 1, 8, 8),
    ).float()

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0
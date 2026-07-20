"""Tests for the U-Net model components."""

import torch
from torch import nn

from src.model import DoubleConv, DownBlock, UNet, UpBlock
from src.reproducibility import seed_everything


def test_double_conv_preserves_spatial_dimensions() -> None:
    """DoubleConv should change channels without changing image size."""

    block = DoubleConv(
        in_channels=1,
        out_channels=16,
    )

    inputs = torch.randn(
        2,
        1,
        128,
        128,
    )

    outputs = block(inputs)

    assert outputs.shape == (
        2,
        16,
        128,
        128,
    )


def test_down_block_halves_spatial_dimensions() -> None:
    """DownBlock should halve height and width."""

    block = DownBlock(
        in_channels=16,
        out_channels=32,
    )

    inputs = torch.randn(
        2,
        16,
        128,
        128,
    )

    outputs = block(inputs)

    assert outputs.shape == (
        2,
        32,
        64,
        64,
    )


def test_up_block_combines_skip_features() -> None:
    """UpBlock should upsample and combine decoder and skip features."""

    block = UpBlock(
        in_channels=64,
        skip_channels=32,
        out_channels=32,
    )

    decoder_inputs = torch.randn(
        2,
        64,
        32,
        32,
    )

    skip_features = torch.randn(
        2,
        32,
        64,
        64,
    )

    outputs = block(
        decoder_inputs,
        skip_features,
    )

    assert outputs.shape == (
        2,
        32,
        64,
        64,
    )


def test_unet_forward_and_backward_pass() -> None:
    """U-Net should produce valid logits and finite gradients."""

    seed_everything(42)

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        dropout_probability=0.20,
    )

    model.train()

    inputs = torch.randn(
        2,
        1,
        128,
        128,
    )

    targets = torch.randint(
        low=0,
        high=2,
        size=(2, 1, 128, 128),
    ).float()

    logits = model(inputs)

    assert logits.shape == targets.shape
    assert torch.isfinite(logits).all()

    loss = nn.BCEWithLogitsLoss()(
        logits,
        targets,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert torch.isfinite(loss)
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        for gradient in gradients
    )

    assert any(
        gradient.abs().sum().item() > 0
        for gradient in gradients
        if gradient is not None
    )
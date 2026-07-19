"""Training utilities for binary medical image segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.metrics import binary_segmentation_metrics_from_logits


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train a segmentation model for one complete epoch."""

    # Enable training-specific behavior such as Dropout
    # and Batch Normalization updates.
    model.train()

    total_loss = 0.0
    processed_samples = 0

    for batch in data_loader:
        images = batch["image"].to(
            device=device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device=device,
            non_blocking=True,
        )

        # Remove gradients left from the previous batch.
        optimizer.zero_grad(
            set_to_none=True,
        )

        # Forward pass: generate one logit for every pixel.
        logits = model(images)

        # Compare model predictions with the true masks.
        loss = criterion(
            logits,
            masks,
        )

        # Backward pass: calculate gradients.
        loss.backward()

        # Update the model parameters using those gradients.
        optimizer.step()

        batch_size = images.shape[0]

        total_loss += (
            loss.detach().item() * batch_size
        )

        processed_samples += batch_size

    if processed_samples == 0:
        raise RuntimeError(
            "Training DataLoader did not provide any samples."
        )

    mean_loss = total_loss / processed_samples

    return {
        "loss": mean_loss,
    }

@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a segmentation model for one complete epoch."""

    # Disable training-specific Dropout behavior
    # and BatchNorm statistic updates.
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    processed_samples = 0

    for batch in data_loader:
        images = batch["image"].to(
            device=device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device=device,
            non_blocking=True,
        )

        # Forward pass only; model parameters are not updated.
        logits = model(images)

        loss = criterion(
            logits,
            masks,
        )

        metrics = binary_segmentation_metrics_from_logits(
            logits=logits,
            targets=masks,
            threshold=threshold,
        )

        batch_size = images.shape[0]

        # Weight batch results by their number of samples.
        total_loss += (
            loss.item() * batch_size
        )

        total_dice += (
            metrics["dice"].item() * batch_size
        )

        total_iou += (
            metrics["iou"].item() * batch_size
        )

        processed_samples += batch_size

    if processed_samples == 0:
        raise RuntimeError(
            "Evaluation DataLoader did not provide any samples."
        )

    return {
        "loss": total_loss / processed_samples,
        "dice": total_dice / processed_samples,
        "iou": total_iou / processed_samples,
    }
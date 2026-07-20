"""Loss functions for binary medical image segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Compute soft Dice loss from segmentation logits."""

    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError(
                "Smooth value must be greater than zero."
            )

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean soft Dice loss for a batch."""

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes: "
                f"{logits.shape} versus {targets.shape}"
            )

        targets = targets.to(
            dtype=logits.dtype,
        )

        probabilities = torch.sigmoid(logits)

        batch_size = logits.shape[0]

        probabilities = probabilities.reshape(
            batch_size,
            -1,
        )

        targets = targets.reshape(
            batch_size,
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice_score.mean()
    
class BCEDiceLoss(nn.Module):
    """Combine binary cross-entropy and soft Dice losses."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if bce_weight < 0:
            raise ValueError(
                "BCE weight must not be negative."
            )

        if dice_weight < 0:
            raise ValueError(
                "Dice weight must not be negative."
            )

        if bce_weight + dice_weight <= 0:
            raise ValueError(
                "At least one loss weight must be greater than zero."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(
            smooth=smooth,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return the weighted BCE and Dice loss combination."""

        bce = self.bce_loss(
            logits,
            targets.to(dtype=logits.dtype),
        )

        dice = self.dice_loss(
            logits,
            targets,
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )
"""U-Net model components for medical image segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Apply two consecutive convolutional blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run the double-convolution block."""

        return self.block(inputs)
    
class DownBlock(nn.Module):
    """Downsample feature maps and apply a double-convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Downsample and process the input feature maps."""

        return self.block(inputs)
    
class UpBlock(nn.Module):
    """Upsample decoder features and combine them with encoder features."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        """Upsample inputs and concatenate the encoder skip features."""

        inputs = self.upsample(inputs)

        if inputs.shape[-2:] != skip_features.shape[-2:]:
            raise ValueError(
                "Spatial dimensions of decoder and skip features "
                f"do not match: {inputs.shape[-2:]} versus "
                f"{skip_features.shape[-2:]}"
            )

        combined = torch.cat(
            [skip_features, inputs],
            dim=1,
        )

        return self.conv(combined)
    
class UNet(nn.Module):
    """Compact U-Net for binary medical image segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout_probability: float = 0.20,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "Input channels must be greater than zero."
            )

        if out_channels <= 0:
            raise ValueError(
                "Output channels must be greater than zero."
            )

        if base_channels <= 0:
            raise ValueError(
                "Base channels must be greater than zero."
            )

        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be in [0, 1)."
            )

        # Encoder
        self.input_block = DoubleConv(
            in_channels=in_channels,
            out_channels=base_channels,
        )

        self.down1 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
        )

        self.down2 = DownBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
        )

        self.down3 = DownBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
        )

        self.bottleneck = DownBlock(
            in_channels=base_channels * 8,
            out_channels=base_channels * 16,
        )

        self.dropout = nn.Dropout2d(
            p=dropout_probability,
        )

        # Decoder
        self.up1 = UpBlock(
            in_channels=base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
        )

        self.up2 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )

        self.up3 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        self.up4 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        self.output_layer = nn.Conv2d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return pixel-wise segmentation logits."""

        skip1 = self.input_block(inputs)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)

        bottleneck = self.bottleneck(skip4)
        bottleneck = self.dropout(bottleneck)

        decoder1 = self.up1(
            bottleneck,
            skip4,
        )

        decoder2 = self.up2(
            decoder1,
            skip3,
        )

        decoder3 = self.up3(
            decoder2,
            skip2,
        )

        decoder4 = self.up4(
            decoder3,
            skip1,
        )

        return self.output_layer(decoder4)
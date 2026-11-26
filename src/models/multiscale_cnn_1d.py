"""Multiscale 1D Convolutional Neural Network architecture for vibration fault classification."""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from src.models.base_model import BaseVibrationClassifier


class MultiscaleTemporalBlock(nn.Module):
    """Parallel temporal convolution block extracting features across multiple time scales.

    Branches:
    - Branch 1 (k=5): Captures short-duration impact transients and high-frequency edge dynamics.
    - Branch 2 (k=15): Captures intermediate localized vibration oscillations (baseline scale).
    - Branch 3 (k=65): Captures wide temporal envelope structures and harmonic modulation patterns.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 16,
        kernel_sizes: Tuple[int, int, int] = (5, 15, 65),
    ) -> None:
        super().__init__()
        self.k1, self.k2, self.k3 = kernel_sizes

        # Branch 1 (Short scale: k=5)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=self.k1, padding=self.k1 // 2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2 (Medium scale: k=15)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=self.k2, padding=self.k2 // 2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 3 (Long scale: k=65)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=self.k3, padding=self.k3 // 2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = branch_channels * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass concatenating multiscale representations along the channel dimension.

        Args:
            x: Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, out_channels, length).
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat([out1, out2, out3], dim=1)

    def forward_branches(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Diagnostic forward pass returning separate branch activations."""
        return {
            f"branch_{self.k1}": self.branch1(x),
            f"branch_{self.k2}": self.branch2(x),
            f"branch_{self.k3}": self.branch3(x),
        }


class MultiscaleVibrationCNN1D(BaseVibrationClassifier):
    """CPU-friendly multiscale 1D CNN with parallel temporal receptive fields.

    Architecture Overview:
    - Stage 1: MultiscaleTemporalBlock with kernels (5, 15, 65) -> MaxPool1d(stride=4).
      Input length 2048 -> 512; channels 1 -> 48.
    - Stage 2: Intermediate Conv1D (k=7) with BatchNorm, ReLU -> MaxPool1d(stride=4).
      Input length 512 -> 128; channels 48 -> 64.
    - Stage 3: Deep Conv1D (k=3) with BatchNorm, ReLU -> MaxPool1d(stride=2).
      Input length 128 -> 64; channels 64 -> 128 (Target layer for Grad-CAM).
    - Stage 4: Adaptive Average Pooling (length=4) -> Flatten (512).
    - Stage 5: MLP Classifier with Dropout (0.3) -> 4-class raw logits.

    Total parameters: ~81,252 (comparable to baseline 73,156).
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 1,
        input_length: int = 2048,
        branch_channels: int = 16,
        multiscale_kernels: Tuple[int, int, int] = (5, 15, 65),
        mid_channels: int = 64,
        deep_channels: int = 128,
        fc_units: int = 64,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes)

        self.input_channels = input_channels
        self.input_length = input_length
        self.dropout_rate = float(dropout_rate)
        self.fc_units = fc_units

        # Stage 1: Multiscale Temporal Block + MaxPool (4)
        self.multiscale_block = MultiscaleTemporalBlock(
            in_channels=input_channels,
            branch_channels=branch_channels,
            kernel_sizes=multiscale_kernels,
        )
        stage1_out_channels = self.multiscale_block.out_channels
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Stage 2: Intermediate Temporal Conv + MaxPool (4)
        self.conv2 = nn.Conv1d(stage1_out_channels, mid_channels, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Stage 3: Deep Feature Conv + MaxPool (2) - Target for Grad-CAM
        self.conv3 = nn.Conv1d(mid_channels, deep_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(deep_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Stage 4: Adaptive Pooling to fixed length 4
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        flattened_dim = deep_channels * 4

        # Stage 5: Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(flattened_dim, self.fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.fc_units, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw class logits.

        Args:
            x: Input vibration tensor of shape (batch_size, 1, 2048).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch_size, 4).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor (batch, channels, length), got {x.ndim}D tensor of shape {x.shape}.")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channel(s), got {x.shape[1]}.")

        # Stage 1: Multiscale feature extraction
        x = self.multiscale_block(x)
        x = self.pool1(x)

        # Stage 2: Intermediate aggregation
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Stage 3: Deep spatial features
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Stage 4: Pooling and classification
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the target convolutional layer (conv3) for Grad-CAM explainability."""
        return self.conv3

    def get_multiscale_block(self) -> MultiscaleTemporalBlock:
        """Return the multiscale temporal block for representation inspection."""
        return self.multiscale_block

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "MultiscaleVibrationCNN1D":
        """Instantiate architecture from configuration dictionary."""
        if config is None:
            from src.utils.config_parser import load_config
            config = load_config()

        m_cfg = config.get("model", {})
        return cls(
            num_classes=m_cfg.get("num_classes", 4),
            input_channels=m_cfg.get("input_channels", 1),
            input_length=m_cfg.get("input_length", 2048),
            branch_channels=16,
            multiscale_kernels=(5, 15, 65),
            mid_channels=64,
            deep_channels=128,
            fc_units=64,
            dropout_rate=m_cfg.get("dropout_rate", 0.3),
        )


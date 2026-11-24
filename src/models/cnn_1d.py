"""1D Convolutional Neural Network (CNN) architecture for vibration fault classification."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from src.models.base_model import BaseVibrationClassifier


class VibrationCNN1D(BaseVibrationClassifier):
    """CPU-optimized 1D CNN for 4-class industrial bearing fault diagnosis.

    Architecture Overview:
    - 3 Conv1D feature extraction blocks with large-to-fine receptive fields:
      Block 1: Conv1D (k=15) -> BatchNorm -> ReLU -> MaxPool (stride=4) [Captures low-freq waves]
      Block 2: Conv1D (k=7)  -> BatchNorm -> ReLU -> MaxPool (stride=4) [Captures transient harmonics]
      Block 3: Conv1D (k=3)  -> BatchNorm -> ReLU -> MaxPool (stride=2) [Captures localized impacts]
    - Adaptive Average Pooling: Decouples feature map length to fixed dimension (4)
    - Regularization: Dropout (default p=0.3) + Batch Normalization
    - Classification Head: Multi-layer Perceptron returning raw class logits

    Input Shape:  (batch_size, 1, 2048)
    Output Shape: (batch_size, 4)
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 1,
        input_length: int = 2048,
        conv_filters: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        pool_sizes: Optional[List[int]] = None,
        fc_units: int = 64,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes)

        self.input_channels = input_channels
        self.input_length = input_length
        self.conv_filters = conv_filters or [32, 64, 128]
        self.kernel_sizes = kernel_sizes or [15, 7, 3]
        self.pool_sizes = pool_sizes or [4, 4, 2]
        self.fc_units = fc_units
        self.dropout_rate = float(dropout_rate)

        if len(self.conv_filters) != 3 or len(self.kernel_sizes) != 3 or len(self.pool_sizes) != 3:
            raise ValueError("conv_filters, kernel_sizes, and pool_sizes must all have length 3.")

        f1, f2, f3 = self.conv_filters
        k1, k2, k3 = self.kernel_sizes
        p1, p2, p3 = self.pool_sizes

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, f1, kernel_size=k1, stride=1, padding=k1 // 2),
            nn.BatchNorm1d(f1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=p1, stride=p1),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(f1, f2, kernel_size=k2, stride=1, padding=k2 // 2),
            nn.BatchNorm1d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=p2, stride=p2),
        )

        # Block 3 (Target layer for Grad-CAM explainability)
        self.conv3 = nn.Conv1d(f2, f3, kernel_size=k3, stride=1, padding=k3 // 2)
        self.bn3 = nn.BatchNorm1d(f3)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=p3, stride=p3)

        # Adaptive spatial pooling to fixed length 4
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        flattened_dim = f3 * 4

        # Regularization & Classification Head
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

        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Spatial pooling and flattening
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Classifier head -> raw logits (no softmax)
        logits = self.classifier(x)
        return logits

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the final convolutional layer (conv3) used for Grad-CAM attributions."""
        return self.conv3

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "VibrationCNN1D":
        """Factory constructor instantiating architecture from project configuration."""
        if config is None:
            from src.utils.config_parser import load_config
            config = load_config()

        m_cfg = config.get("model", {})
        return cls(
            num_classes=m_cfg.get("num_classes", 4),
            input_channels=m_cfg.get("input_channels", 1),
            input_length=m_cfg.get("input_length", 2048),
            conv_filters=m_cfg.get("conv_filters", [32, 64, 128]),
            kernel_sizes=m_cfg.get("kernel_sizes", [15, 7, 3]),
            pool_sizes=m_cfg.get("pool_sizes", [4, 4, 2]),
            fc_units=m_cfg.get("fc_units", [128, 64])[0] if isinstance(m_cfg.get("fc_units"), list) else 64,
            dropout_rate=m_cfg.get("dropout_rate", 0.3),
        )


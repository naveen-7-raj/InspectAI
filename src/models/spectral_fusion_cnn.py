"""Dual-branch 1D CNN fusing raw vibration waveforms with frequency-domain spectral representations."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.base_model import BaseVibrationClassifier


class SpectralFeatureBranch(nn.Module):
    """Convolutional feature extractor for 1,025-bin one-sided spectral representations."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 64,
        in_bins: int = 1025,
        embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if embedding_dim is not None:
            out_features = embedding_dim
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=15, stride=4, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AdaptiveAvgPool1d(4)

        self.fc = nn.Sequential(
            nn.Linear(32 * 4, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting compact spectral embeddings.

        Args:
            x: Spectral tensor of shape (batch_size, 1, 1025) or (batch_size, 1025).

        Returns:
            torch.Tensor: Spectral feature embedding of shape (batch_size, out_features).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)


class SpectralFusionCNN(BaseVibrationClassifier):
    """Dual-branch CNN combining 1D raw waveform representations with spectral representations.

    Branches:
    - Waveform Branch: 3 Conv1D blocks (k=15, 7, 3) identical to baseline feature backbone.
      Extracts temporal shock dynamics into a 512-dim embedding. Target layer: conv3 (Grad-CAM).
    - Spectral Branch: SpectralFeatureBranch extracting 64-dim harmonic/envelope embeddings.
    - Fusion Classifier: Concatenates (512 + 64 = 576) -> MLP (576 -> 64 -> 4) -> 4-class logits.

    Total parameters: ~89,332 (lightweight and CPU-friendly).
    """

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 1,
        waveform_length: int = 2048,
        spectral_bins: int = 1025,
        spectral_type: str = "fft",
        fc_units: int = 64,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes)

        self.input_channels = input_channels
        self.waveform_length = waveform_length
        self.spectral_bins = spectral_bins
        self.spectral_type = spectral_type
        self.dropout_rate = float(dropout_rate)
        self.fc_units = fc_units

        # Waveform Feature Backbone (Matches baseline VibrationCNN1D conv layers)
        self.wave_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.wave_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.wave_pool = nn.AdaptiveAvgPool1d(4)
        wave_embedding_dim = 128 * 4  # 512

        # Spectral Feature Backbone
        self.spectral_branch = SpectralFeatureBranch(in_channels=1, out_features=64)
        spectral_embedding_dim = 64

        # Fusion Classifier Head
        fused_dim = wave_embedding_dim + spectral_embedding_dim  # 576
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(fused_dim, self.fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.fc_units, self.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        spectral_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass supporting combined tensor (B, 1, 3073) or separate inputs.

        Args:
            x: Combined tensor of shape (B, 1, 3073) containing [waveform, spectral],
               or raw waveform tensor of shape (B, 1, 2048).
            spectral_input: Optional spectral tensor of shape (B, 1, 1025) or (B, 1025).

        Returns:
            torch.Tensor: Unnormalized 4-class logits of shape (B, 4).
        """
        if spectral_input is not None:
            wave = x
            spec = spectral_input
        else:
            if x.ndim != 3:
                raise ValueError(f"Expected 3D input tensor (B, 1, L), got shape {x.shape}.")
            if x.shape[-1] == (self.waveform_length + self.spectral_bins):
                # Combined tensor: [:2048] is waveform, [2048:] is spectral
                wave = x[:, :, :self.waveform_length]
                spec = x[:, :, self.waveform_length:]
            elif x.shape[-1] == self.waveform_length:
                raise ValueError("Received waveform-only tensor but spectral_input was not provided.")
            else:
                raise ValueError(
                    f"Unexpected tensor length {x.shape[-1]}. Expected {self.waveform_length + self.spectral_bins}."
                )

        # 1. Waveform Feature Extraction
        w = self.wave_block1(wave)
        w = self.wave_block2(w)
        w = self.conv3(w)
        w = self.bn3(w)
        w = self.relu3(w)
        w = self.pool3(w)
        w = self.wave_pool(w)
        w_emb = torch.flatten(w, 1)  # (B, 512)

        # 2. Spectral Feature Extraction
        s_emb = self.spectral_branch(spec)  # (B, 64)

        # 3. Concatenation and Classification
        fused = torch.cat([w_emb, s_emb], dim=1)  # (B, 576)
        logits = self.classifier(fused)
        return logits

    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the target convolutional layer (conv3) for Grad-CAM explainability."""
        return self.conv3

    def get_cam_target_layer(self) -> nn.Module:
        """Alias for get_target_layer_for_cam."""
        return self.conv3

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        spectral_type: str = "fft",
    ) -> "SpectralFusionCNN":
        """Factory constructor instantiating model from configuration."""
        if config is None:
            from src.utils.config_parser import load_config
            config = load_config()

        m_cfg = config.get("model", {})
        return cls(
            num_classes=m_cfg.get("num_classes", 4),
            input_channels=1,
            waveform_length=2048,
            spectral_bins=1025,
            spectral_type=spectral_type,
            fc_units=64,
            dropout_rate=m_cfg.get("dropout_rate", 0.3),
        )

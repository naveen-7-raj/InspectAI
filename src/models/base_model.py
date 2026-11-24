"""Base model abstract class for vibration fault classifiers."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn


class BaseVibrationClassifier(nn.Module, ABC):
    """Abstract base class for PyTorch vibration fault classification models.

    Concrete implementations will include:
    - VibrationCNN1D: Multi-scale 1D Convolutional Neural Network for raw signals.
    - VibrationCNN2D: 2D Convolutional Neural Network for time-frequency spectrograms.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class logits."""
        pass

    @abstractmethod
    def get_target_layer_for_cam(self) -> nn.Module:
        """Return the target convolutional layer used by Grad-CAM for explainability."""
        pass


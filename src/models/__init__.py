"""Neural network model architectures for vibration signal classification."""

from src.models.base_model import BaseVibrationClassifier
from src.models.cnn_1d import VibrationCNN1D

__all__ = ["BaseVibrationClassifier", "VibrationCNN1D"]

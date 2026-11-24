"""Inference engine for vibration fault diagnosis and window classification."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.normalizer import VibrationNormalizer

logger = logging.getLogger("InspectAI.Predictor")

DEFAULT_CLASS_NAMES = [
    "Normal",
    "Ball Fault",
    "Inner Race Fault",
    "Outer Race Fault",
]


class StreamingPredictor:
    """Performs real-time window classification and inference on CPU.

    Encapsulates trained model execution, feature normalization, and uncalibrated
    softmax probability formatting.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[Sequence[str]] = None,
        normalizer: Optional[VibrationNormalizer] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize StreamingPredictor.

        Args:
            model: Trained 1D CNN (e.g. VibrationCNN1D).
            class_names: List of class names corresponding to output logits.
            normalizer: Optional fitted VibrationNormalizer for raw waveform input.
            device: Computing device ('cpu').
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        self.normalizer = normalizer

    @torch.no_grad()
    def predict_window(
        self,
        window: Union[torch.Tensor, np.ndarray, Sequence[float]],
        is_normalized: bool = False,
    ) -> Dict[str, Any]:
        """Classify a single vibration window of 2,048 samples.

        Args:
            window: 1D vibration array of length 2048 (raw or normalized).
            is_normalized: Whether window is already normalized.

        Returns:
            Dict containing:
                - predicted_class: string label
                - class_id: integer index (0-3)
                - logits: list of raw float logits
                - probabilities: list of uncalibrated softmax scores
                - class_probabilities: dict mapping class name to score
                - confidence_note: reminder that scores are uncalibrated
                - waveform_normalized: 1D numpy array of normalized input
        """
        # Validate and format input
        if isinstance(window, torch.Tensor):
            arr = window.detach().cpu().numpy().squeeze().astype(np.float32)
        else:
            arr = np.asarray(window, dtype=np.float32).squeeze()

        if arr.ndim != 1 or len(arr) != 2048:
            raise ValueError(f"Expected 1D vibration window of 2048 samples, got shape {arr.shape}.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Input vibration window contains NaN or infinite values.")

        # Apply normalization
        if is_normalized:
            norm_arr = arr
        else:
            if self.normalizer is not None and self.normalizer.is_fitted:
                norm_arr = self.normalizer.transform(arr)
            else:
                norm_arr = arr.copy()

        # Convert to tensor: (1, 1, 2048)
        tensor = torch.from_numpy(norm_arr).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Forward pass
        logits_tensor = self.model(tensor)

        if not torch.isfinite(logits_tensor).all():
            raise RuntimeError("Non-finite logits produced during model forward pass.")

        probs_tensor = F.softmax(logits_tensor, dim=1)

        raw_logits = logits_tensor[0].cpu().numpy().astype(float).tolist()
        probs = probs_tensor[0].cpu().numpy().astype(float).tolist()
        pred_idx = int(np.argmax(raw_logits))

        class_prob_map = {
            name: round(probs[i], 6) for i, name in enumerate(self.class_names)
        }

        return {
            "predicted_class": self.class_names[pred_idx],
            "class_id": pred_idx,
            "logits": [round(val, 6) for val in raw_logits],
            "probabilities": [round(val, 6) for val in probs],
            "class_probabilities": class_prob_map,
            "confidence_note": "Softmax scores are uncalibrated model outputs and do not represent true empirical probabilities.",
            "waveform_normalized": norm_arr,
        }

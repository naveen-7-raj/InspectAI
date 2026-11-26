"""Explainable AI (XAI) engine using 1D Grad-CAM for vibration fault diagnosis."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.normalizer import VibrationNormalizer

logger = logging.getLogger("InspectAI.Explainer")

DEFAULT_CLASS_NAMES = [
    "Normal",
    "Ball Fault",
    "Inner Race Fault",
    "Outer Race Fault",
]


class VibrationExplainer:
    """Generates 1D Grad-CAM attributions for 1D CNN vibration fault predictions.

    Mathematical Formulation:
    1. Forward Pass: Computes activations A_k(t) at target convolutional layer (conv3).
    2. Backward Pass: Computes gradients d(y^c) / d(A_k(t)) for class c.
    3. Global Average Pooling: alpha_k = (1 / T) * sum_t (d(y^c) / d(A_k(t))).
    4. Weighted Combination: L_GradCAM^c = ReLU(sum_k (alpha_k * A_k(t))).
    5. 1D Interpolation: Upsamples heatmap from target feature length (128) to input length (2048).
    6. Normalization: Min-Max scales heatmap to [0.0, 1.0] with zero-division protection.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        normalizer: Optional[VibrationNormalizer] = None,
        spectral_normalizer: Optional[Any] = None,
        class_names: Optional[List[str]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize VibrationExplainer.

        Args:
            model: Trained 1D CNN (e.g. VibrationCNN1D or SpectralFusionCNN).
            target_layer: Convolutional layer to explain. Defaults to model.get_target_layer_for_cam() or model.conv3.
            normalizer: Optional fitted VibrationNormalizer for raw waveform inputs.
            spectral_normalizer: Optional fitted SpectralNormalizer for Hilbert envelope spectrum inputs.
            class_names: Ordered class names list.
            device: Computing device ('cpu').
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer_for_cam") and callable(model.get_target_layer_for_cam):
            self.target_layer = model.get_target_layer_for_cam()
        elif hasattr(model, "conv3"):
            self.target_layer = model.conv3
        else:
            # Fallback: locate the last nn.Conv1d module
            conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
            if not conv_modules:
                raise ValueError("Could not find any nn.Conv1d module in the provided model.")
            self.target_layer = conv_modules[-1]

        self.normalizer = normalizer
        self.spectral_normalizer = spectral_normalizer
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    def compute_gradcam(
        self,
        input_tensor: Union[torch.Tensor, np.ndarray],
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
        """Compute 1D Grad-CAM activation heatmap across temporal input window.

        Args:
            input_tensor: Normalized vibration tensor of shape (1, 1, 2048) or (1, 2048) or (2048,).
            target_class: Target class index to explain (0-3). If None, explains predicted class.

        Returns:
            Tuple containing:
                - heatmap: 1D NumPy array of length 2048 in range [0, 1].
                - predicted_class: Index of predicted class.
                - target_class: Index of explained class.
                - logits: 1D NumPy array of raw logits.
                - probabilities: 1D NumPy array of uncalibrated softmax probabilities.
        """
        # Ensure 3D torch tensor: (1, 1, 2048)
        if isinstance(input_tensor, np.ndarray):
            tensor = torch.from_numpy(input_tensor.astype(np.float32))
        else:
            tensor = input_tensor.clone().detach().float()

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 3:
            raise ValueError(f"Expected 1D, 2D, or 3D input tensor, got {tensor.ndim}D.")

        tensor = tensor.to(self.device)
        seq_len = tensor.size(2)

        # Hook storage containers
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            activations.append(out)

        def backward_hook(module: nn.Module, grad_inp: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
            gradients.append(grad_out[0])

        fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

        is_dual = hasattr(self.model, "spectral_branch") or hasattr(self.model, "spectral_bins")

        try:
            self.model.zero_grad()

            # Forward pass (gradient tracking enabled for Grad-CAM)
            if is_dual and tensor.size(2) == 2048:
                arr_np = tensor.detach().cpu().numpy().squeeze()
                from src.features.frequency_features import compute_hilbert_envelope_spectrum
                _, env_mag, _ = compute_hilbert_envelope_spectrum(arr_np, sampling_rate_hz=12000)
                if self.spectral_normalizer is not None and getattr(self.spectral_normalizer, "is_fitted", False):
                    norm_spec = self.spectral_normalizer.transform(env_mag)
                else:
                    norm_spec = env_mag
                s_tensor = torch.from_numpy(norm_spec).float().unsqueeze(0).unsqueeze(0).to(self.device)
                logits = self.model(tensor, spectral_input=s_tensor)
            else:
                logits = self.model(tensor)

            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits encountered during Grad-CAM forward pass.")

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            raw_logits = logits.detach().cpu().numpy()[0]
            pred_class = int(np.argmax(raw_logits))

            # Select target class
            if target_class is None:
                class_to_explain = pred_class
            else:
                class_to_explain = int(target_class)

            if class_to_explain < 0 or class_to_explain >= logits.size(1):
                raise ValueError(f"Target class {class_to_explain} out of bounds for {logits.size(1)} classes.")

            # Backward pass for target class logit
            score = logits[0, class_to_explain]
            score.backward(retain_graph=False)

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

            act = activations[0]   # Shape: (1, C, T) e.g. (1, 128, 128)
            grad = gradients[0]   # Shape: (1, C, T) e.g. (1, 128, 128)

            # Global Average Pooling of gradients across temporal dimension T
            # alpha_k shape: (1, C, 1)
            alpha_k = torch.mean(grad, dim=2, keepdim=True)

            # Weighted linear combination: sum_k alpha_k * A_k
            # cam shape: (1, 1, T)
            cam = torch.sum(alpha_k * act, dim=1, keepdim=True)

            # ReLU to keep features with positive influence on target class score
            cam = F.relu(cam)

            # Upsample linearly to original sequence length (2048)
            target_up_len = 2048 if (is_dual and seq_len in (2048, 3073)) else seq_len
            cam_upsampled = F.interpolate(cam, size=target_up_len, mode="linear", align_corners=False)
            heatmap = cam_upsampled.squeeze().detach().cpu().numpy()

            # Normalize heatmap safely to [0, 1]
            h_min, h_max = float(heatmap.min()), float(heatmap.max())
            denom = h_max - h_min
            if denom > 1e-8:
                heatmap = (heatmap - h_min) / denom
            else:
                heatmap = np.zeros_like(heatmap, dtype=np.float32)

            return heatmap, pred_class, class_to_explain, raw_logits, probs

        finally:
            # Clean up hooks and zero gradients so model state is completely invariant
            fwd_handle.remove()
            bwd_handle.remove()
            self.model.zero_grad()

    def explain_window(
        self,
        vibration_window: Union[torch.Tensor, np.ndarray],
        target_class: Optional[Union[int, str]] = None,
        is_normalized: bool = False,
        sampling_rate_hz: int = 12000,
        importance_threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """Comprehensive explanation interface for streaming or offline vibration windows.

        Args:
            vibration_window: 1D vibration array of length 2048.
            target_class: Optional class index (0-3) or class name to explain.
            is_normalized: Whether vibration_window is already normalized.
            sampling_rate_hz: Sampling frequency in Hz for timestamp conversion.
            importance_threshold: Heatmap value threshold defining 'important regions'.

        Returns:
            Dict containing:
                - predicted_class_id
                - predicted_class_name
                - target_class_id
                - target_class_name
                - logits
                - probabilities (uncalibrated model softmax probabilities)
                - waveform_raw
                - waveform_normalized
                - gradcam_heatmap (length 2048 array in [0, 1])
                - important_regions (list of start/end intervals where heatmap > threshold)
                - timestamps_sec
        """
        if isinstance(vibration_window, torch.Tensor):
            arr = vibration_window.detach().cpu().numpy().squeeze().astype(np.float32)
        else:
            arr = np.asarray(vibration_window, dtype=np.float32).squeeze()

        if arr.ndim != 1 or len(arr) != 2048:
            raise ValueError(f"Expected 1D vibration window of 2048 samples, got shape {arr.shape}.")

        # Handle Normalization
        if is_normalized:
            waveform_norm = arr
            if self.normalizer is not None and self.normalizer.is_fitted:
                waveform_raw = self.normalizer.inverse_transform(arr)
            else:
                waveform_raw = arr.copy()
        else:
            waveform_raw = arr
            if self.normalizer is not None and self.normalizer.is_fitted:
                waveform_norm = self.normalizer.transform(arr)
            else:
                waveform_norm = arr.copy()

        # Resolve target class index
        target_idx: Optional[int] = None
        if target_class is not None:
            if isinstance(target_class, str):
                clean_target = target_class.strip().lower().replace("_", " ")
                for i, cname in enumerate(self.class_names):
                    if cname.lower().replace("_", " ") == clean_target:
                        target_idx = i
                        break
                if target_idx is None:
                    raise ValueError(f"Target class name '{target_class}' not found in {self.class_names}.")
            else:
                target_idx = int(target_class)

        # Compute Grad-CAM
        heatmap, pred_id, expl_id, logits, probs = self.compute_gradcam(
            input_tensor=waveform_norm,
            target_class=target_idx,
        )

        # Extract high-importance temporal segments
        important_regions = self._extract_important_regions(
            heatmap=heatmap,
            threshold=importance_threshold,
            sampling_rate_hz=sampling_rate_hz,
        )

        timestamps = np.arange(len(arr)) / float(sampling_rate_hz)

        return {
            "predicted_class_id": pred_id,
            "predicted_class_name": self.class_names[pred_id],
            "target_class_id": expl_id,
            "target_class_name": self.class_names[expl_id],
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "confidence_note": "Probabilities are uncalibrated model softmax values.",
            "waveform_raw": waveform_raw,
            "waveform_normalized": waveform_norm,
            "gradcam_heatmap": heatmap,
            "important_regions": important_regions,
            "timestamps_sec": timestamps,
            "sampling_rate_hz": sampling_rate_hz,
            "target_layer_name": self.target_layer.__class__.__name__,
        }

    @staticmethod
    def _extract_important_regions(
        heatmap: np.ndarray,
        threshold: float = 0.6,
        sampling_rate_hz: int = 12000,
    ) -> List[Dict[str, Any]]:
        """Identify contiguous temporal segments where Grad-CAM attribution exceeds threshold."""
        mask = heatmap >= threshold
        regions = []
        in_region = False
        start_idx = 0

        for idx, val in enumerate(mask):
            if val and not in_region:
                in_region = True
                start_idx = idx
            elif not val and in_region:
                in_region = False
                end_idx = idx - 1
                peak_val = float(np.max(heatmap[start_idx : end_idx + 1]))
                regions.append({
                    "start_sample": start_idx,
                    "end_sample": end_idx,
                    "start_time_sec": round(start_idx / sampling_rate_hz, 5),
                    "end_time_sec": round(end_idx / sampling_rate_hz, 5),
                    "peak_importance": round(peak_val, 4),
                    "duration_samples": end_idx - start_idx + 1,
                })

        if in_region:
            end_idx = len(heatmap) - 1
            peak_val = float(np.max(heatmap[start_idx : end_idx + 1]))
            regions.append({
                "start_sample": start_idx,
                "end_sample": end_idx,
                "start_time_sec": round(start_idx / sampling_rate_hz, 5),
                "end_time_sec": round(end_idx / sampling_rate_hz, 5),
                "peak_importance": round(peak_val, 4),
                "duration_samples": end_idx - start_idx + 1,
            })

        return regions

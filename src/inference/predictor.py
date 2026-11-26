"""Inference engine for vibration fault diagnosis and window classification."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.frequency_features import (
    SpectralNormalizer,
    compute_hilbert_envelope_spectrum,
)
from src.inference.domain_adapter import DomainAdapter, AdapterStatus
from src.models.spectral_fusion_cnn import SpectralFusionCNN
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

    Encapsulates trained model execution, feature normalization, optional Hilbert
    envelope-spectrum feature extraction for dual-branch architectures, and uncalibrated
    softmax probability formatting.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[Sequence[str]] = None,
        normalizer: Optional[VibrationNormalizer] = None,
        spectral_normalizer: Optional[SpectralNormalizer] = None,
        sampling_rate_hz: int = 12000,
        device: Union[str, torch.device] = "cpu",
        domain_adapter: Optional[DomainAdapter] = None,
    ) -> None:
        """Initialize StreamingPredictor.

        Args:
            model: Trained model (e.g. VibrationCNN1D or SpectralFusionCNN).
            class_names: List of class names corresponding to output logits.
            normalizer: Optional fitted VibrationNormalizer for raw waveform input.
            spectral_normalizer: Optional fitted SpectralNormalizer for Hilbert envelope spectrum.
            sampling_rate_hz: Signal sampling rate (default 12,000 Hz).
            device: Computing device ('cpu').
            domain_adapter: Optional DomainAdapter instance.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        self.normalizer = normalizer
        self.spectral_normalizer = spectral_normalizer
        self.sampling_rate_hz = sampling_rate_hz
        self.is_dual_branch = hasattr(self.model, "spectral_branch") or isinstance(self.model, SpectralFusionCNN)
        self.domain_adapter = domain_adapter

    @torch.no_grad()
    def extract_features(
        self,
        window: Union[torch.Tensor, np.ndarray, Sequence[float]],
        is_normalized: bool = False,
    ) -> np.ndarray:
        """Extract 576-dim fused feature embeddings (512-dim wave + 64-dim spectral)."""
        if isinstance(window, torch.Tensor):
            arr = window.detach().cpu().numpy().squeeze().astype(np.float32)
        else:
            arr = np.asarray(window, dtype=np.float32).squeeze()

        if is_normalized:
            norm_arr = arr
        else:
            if self.normalizer is not None and self.normalizer.is_fitted:
                norm_arr = self.normalizer.transform(arr)
            else:
                norm_arr = arr.copy()

        w_tensor = torch.from_numpy(norm_arr).float().unsqueeze(0).unsqueeze(0).to(self.device)

        if self.is_dual_branch:
            _, env_mag, _ = compute_hilbert_envelope_spectrum(arr, sampling_rate_hz=self.sampling_rate_hz)
            if self.spectral_normalizer is not None and self.spectral_normalizer.is_fitted:
                norm_spec = self.spectral_normalizer.transform(env_mag)
            else:
                norm_spec = env_mag
            s_tensor = torch.from_numpy(norm_spec).float().unsqueeze(0).unsqueeze(0).to(self.device)

            w = self.model.wave_block1(w_tensor)
            w = self.model.wave_block2(w)
            w = self.model.conv3(w)
            w = self.model.bn3(w)
            w = self.model.relu3(w)
            w = self.model.pool3(w)
            w = self.model.wave_pool(w)
            w_emb = torch.flatten(w, 1)

            s_emb = self.model.spectral_branch(s_tensor)
            fused = torch.cat([w_emb, s_emb], dim=1)
            return fused.cpu().numpy().squeeze(0)
        else:
            w = self.model.wave_block1(w_tensor)
            w = self.model.wave_block2(w)
            w = self.model.conv3(w)
            w = self.model.bn3(w)
            w = self.model.relu3(w)
            w = self.model.pool3(w)
            w = self.model.wave_pool(w)
            return torch.flatten(w, 1).cpu().numpy().squeeze(0)

    @torch.no_grad()
    def predict_window(
        self,
        window: Union[torch.Tensor, np.ndarray, Sequence[float]],
        is_normalized: bool = False,
        mode: str = "baseline",
    ) -> Dict[str, Any]:
        """Classify a single vibration window of 2,048 samples.

        Args:
            window: 1D vibration array of length 2048 (raw or normalized).
            is_normalized: Whether waveform is already normalized.
            mode: Inference mode: 'baseline' or 'adaptive'.

        Returns:
            Dict containing classification predictions, probabilities, and adapter telemetry.
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

        # Apply waveform normalization
        if is_normalized:
            norm_arr = arr
        else:
            if self.normalizer is not None and self.normalizer.is_fitted:
                norm_arr = self.normalizer.transform(arr)
            else:
                norm_arr = arr.copy()

        # Convert waveform to tensor: (1, 1, 2048)
        w_tensor = torch.from_numpy(norm_arr).float().unsqueeze(0).unsqueeze(0).to(self.device)

        fused_feat = None
        # Forward pass
        if self.is_dual_branch:
            # Compute Hilbert envelope spectrum on raw/cleaned window
            _, env_mag, _ = compute_hilbert_envelope_spectrum(arr, sampling_rate_hz=self.sampling_rate_hz)
            if self.spectral_normalizer is not None and self.spectral_normalizer.is_fitted:
                norm_spec = self.spectral_normalizer.transform(env_mag)
            else:
                norm_spec = env_mag

            s_tensor = torch.from_numpy(norm_spec).float().unsqueeze(0).unsqueeze(0).to(self.device)

            w = self.model.wave_block1(w_tensor)
            w = self.model.wave_block2(w)
            w = self.model.conv3(w)
            w = self.model.bn3(w)
            w = self.model.relu3(w)
            w = self.model.pool3(w)
            w = self.model.wave_pool(w)
            w_emb = torch.flatten(w, 1)

            s_emb = self.model.spectral_branch(s_tensor)
            fused_feat = torch.cat([w_emb, s_emb], dim=1)
            logits_tensor_baseline = self.model.classifier(fused_feat)
        else:
            logits_tensor_baseline = self.model(w_tensor)

        if not torch.isfinite(logits_tensor_baseline).all():
            raise RuntimeError("Non-finite logits produced during model forward pass.")

        probs_tensor_baseline = F.softmax(logits_tensor_baseline, dim=1)
        raw_logits_base = logits_tensor_baseline[0].cpu().numpy().astype(float).tolist()
        probs_base = probs_tensor_baseline[0].cpu().numpy().astype(float).tolist()
        pred_idx_base = int(np.argmax(raw_logits_base))
        base_prob_map = {name: round(probs_base[i], 6) for i, name in enumerate(self.class_names)}

        # Adaptive Inference Execution
        is_adapted = False
        exec_mode = mode.lower()
        if exec_mode == "adaptive" and fused_feat is not None:
            if self.domain_adapter is not None and self.domain_adapter.is_fitted:
                f_np = fused_feat.cpu().numpy()
                f_aligned = self.domain_adapter.transform(f_np)
                f_tensor = torch.from_numpy(f_aligned).float().to(self.device)
                logits_adapted = self.model.classifier(f_tensor)
                probs_adapted = F.softmax(logits_adapted, dim=1)

                raw_logits = logits_adapted[0].cpu().numpy().astype(float).tolist()
                probs = probs_adapted[0].cpu().numpy().astype(float).tolist()
                pred_idx = int(np.argmax(raw_logits))
                class_prob_map = {name: round(probs[i], 6) for i, name in enumerate(self.class_names)}
                is_adapted = True
            else:
                raw_logits = raw_logits_base
                probs = probs_base
                pred_idx = pred_idx_base
                class_prob_map = base_prob_map
                exec_mode = "adaptive_fallback"
        else:
            raw_logits = raw_logits_base
            probs = probs_base
            pred_idx = pred_idx_base
            class_prob_map = base_prob_map

        adapter_telemetry = self.domain_adapter.get_telemetry() if self.domain_adapter is not None else None

        return {
            "predicted_class": self.class_names[pred_idx],
            "class_id": pred_idx,
            "logits": [round(val, 6) for val in raw_logits],
            "probabilities": [round(val, 6) for val in probs],
            "class_probabilities": class_prob_map,
            "inference_mode": exec_mode,
            "is_adapted": is_adapted,
            "baseline_predicted_class": self.class_names[pred_idx_base],
            "baseline_probabilities": base_prob_map,
            "adapter_telemetry": adapter_telemetry,
            "confidence_note": "Softmax scores are uncalibrated model outputs and do not represent true empirical probabilities.",
            "waveform_normalized": norm_arr,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        windows: Union[torch.Tensor, np.ndarray],
        is_normalized: bool = False,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of 2,048-sample vibration windows."""
        if isinstance(windows, torch.Tensor):
            arrs = windows.detach().cpu().numpy().astype(np.float32)
        else:
            arrs = np.asarray(windows, dtype=np.float32)

        if arrs.ndim != 2 or arrs.shape[1] != 2048:
            raise ValueError(f"Expected 2D batch of shape (N, 2048), got shape {arrs.shape}.")

        if not np.all(np.isfinite(arrs)):
            raise ValueError("Input batch contains NaN or infinite values.")

        if is_normalized:
            norm_arrs = arrs
        else:
            if self.normalizer is not None and self.normalizer.is_fitted:
                norm_arrs = self.normalizer.transform(arrs)
            else:
                norm_arrs = arrs.copy()

        w_tensor = torch.from_numpy(norm_arrs).float().unsqueeze(1).to(self.device)

        if self.is_dual_branch:
            _, env_mags, _ = compute_hilbert_envelope_spectrum(arrs, sampling_rate_hz=self.sampling_rate_hz)
            if self.spectral_normalizer is not None and self.spectral_normalizer.is_fitted:
                norm_specs = self.spectral_normalizer.transform(env_mags)
            else:
                norm_specs = env_mags
            s_tensor = torch.from_numpy(norm_specs).float().unsqueeze(1).to(self.device)
            logits_tensor = self.model(w_tensor, spectral_input=s_tensor)
        else:
            logits_tensor = self.model(w_tensor)

        probs_tensor = F.softmax(logits_tensor, dim=1)
        raw_logits = logits_tensor.cpu().numpy().astype(float)
        probs = probs_tensor.cpu().numpy().astype(float)
        pred_indices = np.argmax(raw_logits, axis=1)

        results = []
        for i in range(len(arrs)):
            p_idx = int(pred_indices[i])
            results.append({
                "predicted_class": self.class_names[p_idx],
                "class_id": p_idx,
                "logits": [round(float(v), 6) for v in raw_logits[i]],
                "probabilities": [round(float(v), 6) for v in probs[i]],
                "class_probabilities": {
                    name: round(float(probs[i, c]), 6) for c, name in enumerate(self.class_names)
                },
            })
        return results

    @classmethod
    def from_final_model(
        cls,
        model_dir: Union[str, Path] = "models/final",
        device: Union[str, torch.device] = "cpu",
        domain_adapter: Optional[DomainAdapter] = None,
    ) -> "StreamingPredictor":
        """Instantiate StreamingPredictor loading from the frozen final release directory."""
        dir_path = Path(model_dir).resolve()
        ckpt_path = dir_path / "inspectai_final_model.pth"
        norm_path = dir_path / "inspectai_final_normalizer.json"
        spec_norm_path = dir_path / "inspectai_final_spectral_normalizer.json"

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Final model checkpoint not found at: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cls_name = ckpt.get("model_class", "SpectralFusionCNN")

        if model_cls_name == "SpectralFusionCNN":
            model = SpectralFusionCNN(
                num_classes=ckpt.get("num_classes", 4),
                waveform_length=ckpt.get("waveform_length", 2048),
                spectral_bins=ckpt.get("spectral_bins", 1025),
                fc_units=ckpt.get("fc_units", 64),
                dropout_rate=ckpt.get("dropout_rate", 0.3),
            )
        else:
            from src.models.cnn_1d import VibrationCNN1D
            model = VibrationCNN1D(fc_units=ckpt.get("fc_units", 64))

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        normalizer = VibrationNormalizer.load(norm_path) if norm_path.is_file() else None
        spec_normalizer = SpectralNormalizer.load(spec_norm_path) if spec_norm_path.is_file() else None

        return cls(
            model=model,
            class_names=ckpt.get("class_names", DEFAULT_CLASS_NAMES),
            normalizer=normalizer,
            spectral_normalizer=spec_normalizer,
            device=device,
            domain_adapter=domain_adapter,
        )

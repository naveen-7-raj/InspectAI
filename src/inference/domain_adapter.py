"""Production-ready unsupervised domain adapter (CORAL) for industrial vibration streaming.

Implements second-order covariance alignment between source machinery distribution (CWRU)
and unlabelled target operational data (e.g. Paderborn or live plant machinery) without modifying
frozen neural network weights or requiring ground-truth fault labels.
"""

from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.linalg as la
import torch

logger = logging.getLogger("InspectAI.DomainAdapter")


class AdapterStatus(str, Enum):
    """Operational status of domain adapter."""
    UNINITIALIZED = "UNINITIALIZED"
    CALIBRATING = "CALIBRATING"
    READY = "READY"
    FALLBACK = "FALLBACK"


class DomainAdapter:
    """Unsupervised Correlation Alignment (CORAL) adapter for vibration feature embeddings.
    
    Transforms target latent features so their second-order statistics match source reference domain:
        X_aligned = (X_target - mu_target) @ (C_target^(-1/2) @ C_source^(1/2)) + mu_source
    
    Guarantees:
    1. Zero model weight modification (classification head and feature extractors stay frozen).
    2. Strictly label-free (never accesses target fault labels).
    3. Fail-safe fallback to unadapted baseline when calibration is insufficient or unstable.
    """

    def __init__(
        self,
        source_mean: Optional[np.ndarray] = None,
        source_cov_sqrt: Optional[np.ndarray] = None,
        min_calibration_samples: int = 50,
        regularization: float = 1e-3,
        max_condition_number: float = 1e8,
    ) -> None:
        """Initialize DomainAdapter.
        
        Args:
            source_mean: Source domain mean vector (dimension d=576).
            source_cov_sqrt: Matrix square root of source covariance C_S^(1/2) (d x d).
            min_calibration_samples: Minimum unlabelled windows required to fit covariance.
            regularization: Tikhonov regularization parameter lambda * I.
            max_condition_number: Maximum allowable covariance condition number.
        """
        self.source_mean = np.asarray(source_mean, dtype=np.float32) if source_mean is not None else None
        self.source_cov_sqrt = np.asarray(source_cov_sqrt, dtype=np.float32) if source_cov_sqrt is not None else None
        self.min_calibration_samples = max(2, int(min_calibration_samples))
        self.regularization = float(regularization)
        self.max_condition_number = float(max_condition_number)

        self.calibration_buffer: List[np.ndarray] = []
        self.is_fitted: bool = False
        self.A_coral: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_covariance: Optional[np.ndarray] = None
        self.condition_number: Optional[float] = None
        self.error_message: Optional[str] = None

        if self.source_mean is not None and self.source_cov_sqrt is not None:
            self.status = AdapterStatus.CALIBRATING
        else:
            self.status = AdapterStatus.UNINITIALIZED

    @property
    def calibration_progress(self) -> float:
        """Return calibration progress ratio (0.0 to 1.0)."""
        if self.is_fitted:
            return 1.0
        return min(1.0, len(self.calibration_buffer) / float(self.min_calibration_samples))

    @property
    def buffer_count(self) -> int:
        """Number of unlabelled calibration windows collected."""
        return len(self.calibration_buffer)

    def set_source_statistics(self, source_features: np.ndarray) -> None:
        """Compute and set reference mean and covariance square root from source embeddings.
        
        Args:
            source_features: 2D array of shape (N_source, d) containing source embeddings.
        """
        X_s = np.asarray(source_features, dtype=np.float64)
        if X_s.ndim != 2 or X_s.shape[0] < 2:
            raise ValueError(f"Expected 2D source features with >=2 samples, got shape {X_s.shape}.")

        dim = X_s.shape[1]
        self.source_mean = np.mean(X_s, axis=0)
        X_s_c = X_s - self.source_mean
        C_s = (X_s_c.T @ X_s_c) / (len(X_s) - 1) + self.regularization * np.eye(dim)

        vals_s, vecs_s = la.eigh(C_s)
        vals_s = np.maximum(vals_s, 1e-8)
        self.source_cov_sqrt = (vecs_s @ np.diag(np.sqrt(vals_s)) @ vecs_s.T).astype(np.float32)
        self.source_mean = self.source_mean.astype(np.float32)

        if not self.is_fitted:
            self.status = AdapterStatus.CALIBRATING

    def validate_window_signal(self, window: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate raw vibration window for streaming ingestion.
        
        Checks for invalid length, non-finite values, and flatline/constant sensor failure.
        """
        arr = np.asarray(window, dtype=np.float32).squeeze()
        if arr.ndim != 1 or len(arr) != 2048:
            return False, f"Invalid window length: expected 2048 samples, got {len(arr)}."
        if not np.all(np.isfinite(arr)):
            return False, "Window contains NaN or infinite values."
        std = float(np.std(arr))
        if std < 1e-6:
            return False, f"Dead/constant sensor signal detected (standard deviation={std:.2e} < 1e-6)."
        return True, None

    def collect_calibration_sample(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Collect a single unlabelled 576-dim feature embedding for calibration.
        
        Args:
            feature_vector: 1D array of length d, or 2D array of shape (1, d).
            
        Returns:
            Dict containing buffer count, required count, progress, and ready status.
        """
        feat = np.asarray(feature_vector, dtype=np.float32).squeeze()
        if feat.ndim != 1:
            raise ValueError(f"Expected 1D feature vector, got shape {feat.shape}.")
        if not np.all(np.isfinite(feat)):
            raise ValueError("Feature vector contains NaN or infinite values.")

        self.calibration_buffer.append(feat)

        return {
            "collected_samples": len(self.calibration_buffer),
            "required_samples": self.min_calibration_samples,
            "progress_ratio": self.calibration_progress,
            "is_ready_to_fit": len(self.calibration_buffer) >= self.min_calibration_samples,
            "status": self.status.value,
        }

    def collect_calibration_batch(self, feature_batch: np.ndarray) -> Dict[str, Any]:
        """Collect multiple unlabelled feature embeddings."""
        batch = np.asarray(feature_batch, dtype=np.float32)
        if batch.ndim != 2:
            raise ValueError(f"Expected 2D feature batch, got shape {batch.shape}.")
        for feat in batch:
            self.collect_calibration_sample(feat)
        return {
            "collected_samples": len(self.calibration_buffer),
            "required_samples": self.min_calibration_samples,
            "progress_ratio": self.calibration_progress,
            "is_ready_to_fit": len(self.calibration_buffer) >= self.min_calibration_samples,
            "status": self.status.value,
        }

    def fit(self, calibration_features: Optional[np.ndarray] = None) -> bool:
        """Fit CORAL operator A = C_T^(-1/2) @ C_S^(1/2) on unlabelled calibration features.
        
        Returns:
            True if fitting succeeded; False if fallback was triggered.
        """
        if self.source_mean is None or self.source_cov_sqrt is None:
            self.status = AdapterStatus.FALLBACK
            self.error_message = "Source domain reference statistics not initialized."
            logger.error(self.error_message)
            return False

        if calibration_features is not None:
            X_t = np.asarray(calibration_features, dtype=np.float64)
        else:
            if len(self.calibration_buffer) < self.min_calibration_samples:
                self.status = AdapterStatus.FALLBACK
                self.error_message = (
                    f"Insufficient calibration samples: collected {len(self.calibration_buffer)} / "
                    f"{self.min_calibration_samples} required windows."
                )
                logger.warning(self.error_message)
                return False
            X_t = np.array(self.calibration_buffer, dtype=np.float64)

        if X_t.shape[0] < 2:
            self.status = AdapterStatus.FALLBACK
            self.error_message = "Calibration dataset must have at least 2 samples."
            logger.error(self.error_message)
            return False

        dim = X_t.shape[1]
        if dim != len(self.source_mean):
            self.status = AdapterStatus.FALLBACK
            self.error_message = f"Feature dimension mismatch: source={len(self.source_mean)}, target={dim}."
            logger.error(self.error_message)
            return False

        # Check for non-finite values
        if not np.all(np.isfinite(X_t)):
            self.status = AdapterStatus.FALLBACK
            self.error_message = "Calibration features contain NaN or infinite values."
            logger.error(self.error_message)
            return False

        # Check for flatline / zero-variance features across all channels
        total_var = float(np.sum(np.var(X_t, axis=0)))
        if total_var < 1e-6:
            self.status = AdapterStatus.FALLBACK
            self.error_message = f"Calibration features have near-zero total variance ({total_var:.2e})."
            logger.error(self.error_message)
            return False

        try:
            mu_t = np.mean(X_t, axis=0)
            X_t_c = X_t - mu_t
            C_t = (X_t_c.T @ X_t_c) / (len(X_t) - 1) + self.regularization * np.eye(dim)

            # Eigendecomposition
            vals_t, vecs_t = la.eigh(C_t)
            vals_t = np.maximum(vals_t, 1e-8)
            cond_num = float(np.max(vals_t) / np.min(vals_t))
            self.condition_number = cond_num

            if cond_num > self.max_condition_number:
                logger.warning(f"Target covariance condition number ({cond_num:.2e}) exceeds limit ({self.max_condition_number:.2e}).")

            C_t_inv_sqrt = vecs_t @ np.diag(1.0 / np.sqrt(vals_t)) @ vecs_t.T
            A_coral = C_t_inv_sqrt @ self.source_cov_sqrt

            self.A_coral = A_coral.astype(np.float32)
            self.target_mean = mu_t.astype(np.float32)
            self.target_covariance = C_t.copy()
            self.is_fitted = True
            self.status = AdapterStatus.READY
            self.error_message = None
            logger.info(f"DomainAdapter calibrated successfully on {len(X_t)} windows (dim={dim}, cond={cond_num:.2f}).")
            return True

        except Exception as exc:
            self.is_fitted = False
            self.status = AdapterStatus.FALLBACK
            self.error_message = f"Numerical error during CORAL operator fitting: {str(exc)}"
            logger.exception(self.error_message)
            return False

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Align target features into source domain embedding space.
        
        If adapter is not ready, raises RuntimeError. Callers should check is_fitted or status.
        """
        if not self.is_fitted or self.A_coral is None or self.target_mean is None or self.source_mean is None:
            raise RuntimeError(f"DomainAdapter cannot transform: adapter status is {self.status.value}. Error: {self.error_message}")

        feats = np.asarray(features, dtype=np.float32)
        is_1d = (feats.ndim == 1)
        if is_1d:
            feats = feats.reshape(1, -1)

        aligned = (feats - self.target_mean) @ self.A_coral + self.source_mean
        return aligned.squeeze(0) if is_1d else aligned

    def reset(self) -> None:
        """Reset calibration buffer and adapter state for new operational stream."""
        self.calibration_buffer.clear()
        self.is_fitted = False
        self.A_coral = None
        self.target_mean = None
        self.condition_number = None
        self.error_message = None
        if self.source_mean is not None and self.source_cov_sqrt is not None:
            self.status = AdapterStatus.CALIBRATING
        else:
            self.status = AdapterStatus.UNINITIALIZED
        logger.info("DomainAdapter reset to initial calibration state.")

    def get_telemetry(self) -> Dict[str, Any]:
        """Return diagnostic telemetry for monitoring and dashboard display."""
        return {
            "status": self.status.value,
            "is_fitted": self.is_fitted,
            "buffered_samples": len(self.calibration_buffer),
            "required_samples": self.min_calibration_samples,
            "progress_percent": round(self.calibration_progress * 100.0, 1),
            "condition_number": round(self.condition_number, 2) if self.condition_number is not None else None,
            "regularization": self.regularization,
            "error_message": self.error_message,
        }

    @classmethod
    def from_processed_cwru(
        cls,
        cwru_windows_path: Union[str, Path],
        predictor: Any,
        min_calibration_samples: int = 50,
        regularization: float = 1e-3,
    ) -> "DomainAdapter":
        """Factory method: Initialize adapter with source statistics from CWRU development windows."""
        p = Path(cwru_windows_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CWRU segmented windows archive not found at: {p}")

        data = np.load(p)
        windows = data["windows"]

        # Compute envelope spectra and normalize
        from src.features.frequency_features import compute_hilbert_envelope_spectrum
        _, env_specs, _ = compute_hilbert_envelope_spectrum(windows, sampling_rate_hz=predictor.sampling_rate_hz)
        norm_w = predictor.normalizer.transform(windows)
        norm_s = predictor.spectral_normalizer.transform(env_specs)

        # Extract fused feature embeddings (512-dim wave + 64-dim spectral = 576-dim)
        w_tensor = torch.tensor(norm_w, dtype=torch.float32).unsqueeze(1).to(predictor.device)
        s_tensor = torch.tensor(norm_s, dtype=torch.float32).unsqueeze(1).to(predictor.device)

        predictor.model.eval()
        with torch.no_grad():
            w = predictor.model.wave_block1(w_tensor)
            w = predictor.model.wave_block2(w)
            w = predictor.model.conv3(w)
            w = predictor.model.bn3(w)
            w = predictor.model.relu3(w)
            w = predictor.model.pool3(w)
            w = predictor.model.wave_pool(w)
            w_emb = torch.flatten(w, 1)

            s_emb = predictor.model.spectral_branch(s_tensor)
            fused = torch.cat([w_emb, s_emb], dim=1).cpu().numpy()

        adapter = cls(
            min_calibration_samples=min_calibration_samples,
            regularization=regularization,
        )
        adapter.set_source_statistics(fused)
        return adapter

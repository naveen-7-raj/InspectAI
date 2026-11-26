"""Training-only signal normalization for vibration time-series supporting multiple strategies."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch

logger = logging.getLogger("InspectAI.Normalizer")

SUPPORTED_STRATEGIES = {"global", "instance", "none"}


class VibrationNormalizer:
    """Configurable signal normalizer supporting global, instance, and passthrough strategies.

    Strategies:
    - 'global': Computes scalar mean and standard deviation from training data only.
      Ensures zero data leakage from validation or test partitions and preserves relative
      amplitude dynamics across all windows.
    - 'instance': Computes mean and standard deviation independently for each individual
      window (per-sample normalization). Eliminates absolute amplitude bias while protecting
      against zero-variance/constant signals.
    - 'none': Passthrough mode that preserves original physical amplitudes without modification.
    """

    def __init__(self, epsilon: float = 1e-8, strategy: str = "global") -> None:
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}.")
        clean_strategy = str(strategy).lower().strip()
        if clean_strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unrecognized normalization strategy '{strategy}'. Supported: {sorted(list(SUPPORTED_STRATEGIES))}."
            )
        self.epsilon = float(epsilon)
        self.strategy = clean_strategy
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self.is_fitted: bool = False
        self.fit_recording_ids: List[str] = []
        self.fit_sample_count: int = 0

    @staticmethod
    def _validate_data(data: Any, name: str = "data") -> np.ndarray:
        """Validate array numeric type, non-emptiness, and absence of NaN/Inf."""
        if data is None:
            raise ValueError(f"Input '{name}' cannot be None.")

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(f"Input '{name}' has non-numeric dtype '{data.dtype}'.")

        if data.size == 0:
            raise ValueError(f"Input '{name}' is empty (0 samples).")

        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"Input '{name}' contains corrupt values ({nan_count} NaNs, {inf_count} Infs). "
                f"Cannot compute normalization parameters."
            )

        return data

    def fit(
        self,
        windows: np.ndarray,
        source_recording_ids: Optional[Sequence[str]] = None,
    ) -> "VibrationNormalizer":
        """Fit normalization parameters.

        For 'global', calculates scalar mean and std across all training sample points.
        For 'instance' and 'none', records metadata and marks fitted state.

        Args:
            windows: 2D numpy array of shape (N_train, window_size) or 1D array.
            source_recording_ids: Optional list of recording IDs in the training set.

        Returns:
            self: Fitted normalizer instance.
        """
        valid_data = self._validate_data(windows, name="training_windows")
        self.fit_sample_count = len(valid_data) if valid_data.ndim == 2 else 1
        self.fit_recording_ids = sorted(list(set(source_recording_ids))) if source_recording_ids else []

        if self.strategy == "global":
            self.mean = float(np.mean(valid_data))
            self.std = float(np.std(valid_data))
            logger.info(
                f"Fitted VibrationNormalizer(strategy='global') on {self.fit_sample_count} training windows "
                f"(mean={self.mean:+.6e}, std={self.std:.6e}, eps={self.epsilon:.1e}, "
                f"recordings={self.fit_recording_ids})."
            )
        elif self.strategy == "instance":
            self.mean = 0.0
            self.std = 1.0
            logger.info(
                f"Initialized VibrationNormalizer(strategy='instance') for {self.fit_sample_count} training windows "
                f"(eps={self.epsilon:.1e}, recordings={self.fit_recording_ids})."
            )
        elif self.strategy == "none":
            self.mean = 0.0
            self.std = 1.0
            logger.info(
                f"Initialized VibrationNormalizer(strategy='none') for {self.fit_sample_count} training windows "
                f"(recordings={self.fit_recording_ids})."
            )

        self.is_fitted = True
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Apply normalization to input windows according to configured strategy.

        Args:
            windows: 1D or 2D array of vibration windows.

        Returns:
            np.ndarray: Normalized array with same shape and dtype float32.
        """
        if not self.is_fitted:
            raise RuntimeError(f"VibrationNormalizer (strategy='{self.strategy}') must be fitted before transforming data.")

        valid_data = self._validate_data(windows, name="transform_input")

        if self.strategy == "global":
            if self.mean is None or self.std is None:
                raise RuntimeError("VibrationNormalizer with strategy='global' requires fitted mean and std.")
            normalized = (valid_data - self.mean) / (self.std + self.epsilon)
            return normalized.astype(np.float32)

        elif self.strategy == "instance":
            d64 = valid_data.astype(np.float64)
            if d64.ndim == 1:
                w_mean = float(np.mean(d64))
                w_std = float(np.std(d64))
                ptp = float(np.ptp(d64))
                if w_std < self.epsilon or ptp < 1e-5:
                    return np.zeros_like(valid_data, dtype=np.float32)
                return ((d64 - w_mean) / w_std).astype(np.float32)
            else:
                w_mean = np.mean(d64, axis=-1, keepdims=True)
                w_std = np.std(d64, axis=-1, keepdims=True)
                ptp = np.max(d64, axis=-1, keepdims=True) - np.min(d64, axis=-1, keepdims=True)
                zero_mask = (w_std < self.epsilon) | (ptp < 1e-5)
                denom = np.maximum(w_std, self.epsilon)
                normalized = np.where(zero_mask, 0.0, (d64 - w_mean) / denom)
                return normalized.astype(np.float32)

        elif self.strategy == "none":
            return valid_data.astype(np.float32)

        raise RuntimeError(f"Unsupported strategy '{self.strategy}'.")

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization to PyTorch tensor according to configured strategy.

        Args:
            tensor: PyTorch float tensor (e.g. shape (1, L) or (B, 1, L)).

        Returns:
            torch.Tensor: Normalized float32 tensor.
        """
        if not self.is_fitted:
            raise RuntimeError(f"VibrationNormalizer (strategy='{self.strategy}') must be fitted before transforming tensor.")

        if self.strategy == "global":
            if self.mean is None or self.std is None:
                raise RuntimeError("VibrationNormalizer with strategy='global' requires fitted mean and std.")
            return ((tensor - self.mean) / (self.std + self.epsilon)).float()

        elif self.strategy == "instance":
            t64 = tensor.double()
            w_mean = t64.mean(dim=-1, keepdim=True)
            w_std = t64.std(dim=-1, keepdim=True, unbiased=False)
            ptp = t64.max(dim=-1, keepdim=True).values - t64.min(dim=-1, keepdim=True).values
            zero_mask = (w_std < self.epsilon) | (ptp < 1e-5)
            denom = torch.clamp(w_std, min=self.epsilon)
            normalized = (t64 - w_mean) / denom
            return torch.where(zero_mask, torch.zeros_like(t64), normalized).float()

        elif self.strategy == "none":
            return tensor.float()

        raise RuntimeError(f"Unsupported strategy '{self.strategy}'.")

    def fit_transform(
        self,
        windows: np.ndarray,
        source_recording_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Fit parameters on training data and return normalized training data."""
        self.fit(windows, source_recording_ids=source_recording_ids)
        return self.transform(windows)

    def inverse_transform(self, normalized_windows: np.ndarray) -> np.ndarray:
        """Reconstruct original physical amplitude values (in g) when supported.

        Args:
            normalized_windows: Array of normalized samples.

        Returns:
            np.ndarray: Physical vibration acceleration in g.
        """
        if not self.is_fitted:
            raise RuntimeError("VibrationNormalizer must be fitted before inverse transform.")

        valid_data = self._validate_data(normalized_windows, name="inverse_input")

        if self.strategy == "global":
            if self.mean is None or self.std is None:
                raise RuntimeError("VibrationNormalizer with strategy='global' requires fitted mean and std.")
            return (valid_data * (self.std + self.epsilon)) + self.mean
        elif self.strategy == "instance":
            raise RuntimeError(
                "Inverse transform is not supported for strategy='instance' because window-specific "
                "statistics are not preserved globally."
            )
        elif self.strategy == "none":
            return valid_data

        raise RuntimeError(f"Unsupported strategy '{self.strategy}'.")

    def to_dict(self) -> Dict[str, Any]:
        """Export parameters and provenance metadata to a dictionary."""
        return {
            "strategy": self.strategy,
            "mean": self.mean,
            "std": self.std,
            "epsilon": self.epsilon,
            "is_fitted": self.is_fitted,
            "fit_sample_count": self.fit_sample_count,
            "fit_recording_ids": self.fit_recording_ids,
        }

    def save(self, file_path: Union[str, Path]) -> Path:
        """Persist normalization parameters and strategy to a JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted VibrationNormalizer.")

        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved VibrationNormalizer (strategy='{self.strategy}') to: {path}")
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "VibrationNormalizer":
        """Load normalization parameters and strategy from a JSON file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Normalizer parameter file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        strategy = data.get("strategy", "global")
        instance = cls(epsilon=data.get("epsilon", 1e-8), strategy=strategy)
        instance.mean = float(data["mean"]) if data["mean"] is not None else None
        instance.std = float(data["std"]) if data["std"] is not None else None
        instance.is_fitted = bool(data["is_fitted"])
        instance.fit_sample_count = int(data.get("fit_sample_count", 0))
        instance.fit_recording_ids = list(data.get("fit_recording_ids", []))

        logger.info(
            f"Loaded VibrationNormalizer from: {path} (strategy='{instance.strategy}', "
            f"mean={instance.mean if instance.mean is not None else 0.0:.6e}, "
            f"std={instance.std if instance.std is not None else 0.0:.6e})"
        )
        return instance

"""Training-only global signal normalization for vibration time-series."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch

logger = logging.getLogger("InspectAI.Normalizer")


class VibrationNormalizer:
    """Computes global scalar mean and standard deviation from training data only.

    Ensures zero data leakage from validation or test partitions and preserves
    relative amplitude dynamics across all windows.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}.")
        self.epsilon = float(epsilon)
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
        """Calculate global scalar mean and std across all training sample points.

        Args:
            windows: 2D numpy array of shape (N_train, window_size) or 1D array.
            source_recording_ids: Optional list of recording IDs in the training set.

        Returns:
            self: Fitted normalizer instance.
        """
        valid_data = self._validate_data(windows, name="training_windows")

        # Global mean and standard deviation across all points
        self.mean = float(np.mean(valid_data))
        self.std = float(np.std(valid_data))
        self.fit_sample_count = len(valid_data) if valid_data.ndim == 2 else 1
        self.fit_recording_ids = sorted(list(set(source_recording_ids))) if source_recording_ids else []
        self.is_fitted = True

        logger.info(
            f"Fitted VibrationNormalizer on {self.fit_sample_count} training windows "
            f"(mean={self.mean:+.6e}, std={self.std:.6e}, eps={self.epsilon:.1e}, "
            f"recordings={self.fit_recording_ids})."
        )
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Apply fitted global normalization to input windows.

        Args:
            windows: 1D or 2D array of vibration windows.

        Returns:
            np.ndarray: Normalized array with same shape and dtype float32.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("VibrationNormalizer must be fitted before transforming data.")

        valid_data = self._validate_data(windows, name="transform_input")
        # Global z-score standardization with epsilon protection
        normalized = (valid_data - self.mean) / (self.std + self.epsilon)
        return normalized.astype(np.float32)

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply fitted normalization to PyTorch tensor.

        Args:
            tensor: PyTorch float tensor.

        Returns:
            torch.Tensor: Normalized float32 tensor.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("VibrationNormalizer must be fitted before transforming tensor.")

        return ((tensor - self.mean) / (self.std + self.epsilon)).float()

    def fit_transform(
        self,
        windows: np.ndarray,
        source_recording_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Fit parameters on training data and return normalized training data."""
        self.fit(windows, source_recording_ids=source_recording_ids)
        return self.transform(windows)

    def inverse_transform(self, normalized_windows: np.ndarray) -> np.ndarray:
        """Reconstruct original physical amplitude values (in g).

        Args:
            normalized_windows: Array of normalized samples.

        Returns:
            np.ndarray: Physical vibration acceleration in g.
        """
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("VibrationNormalizer must be fitted before inverse transform.")

        valid_data = self._validate_data(normalized_windows, name="inverse_input")
        unnormalized = (valid_data * (self.std + self.epsilon)) + self.mean
        return unnormalized

    def to_dict(self) -> Dict[str, Any]:
        """Export parameters and provenance metadata to a dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "epsilon": self.epsilon,
            "is_fitted": self.is_fitted,
            "fit_sample_count": self.fit_sample_count,
            "fit_recording_ids": self.fit_recording_ids,
        }

    def save(self, file_path: Union[str, Path]) -> Path:
        """Persist fitted normalization parameters to a JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted VibrationNormalizer.")

        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved VibrationNormalizer parameters to: {path}")
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "VibrationNormalizer":
        """Load fitted normalization parameters from a JSON file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Normalizer parameter file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(epsilon=data.get("epsilon", 1e-8))
        instance.mean = float(data["mean"])
        instance.std = float(data["std"])
        instance.is_fitted = bool(data["is_fitted"])
        instance.fit_sample_count = int(data.get("fit_sample_count", 0))
        instance.fit_recording_ids = list(data.get("fit_recording_ids", []))

        logger.info(f"Loaded VibrationNormalizer from: {path} (mean={instance.mean:.6e}, std={instance.std:.6e})")
        return instance


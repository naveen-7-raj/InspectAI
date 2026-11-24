"""PyTorch Dataset implementation for CWRU vibration signal windows."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.training.dataset_splitter import DatasetSplitter
from src.training.normalizer import VibrationNormalizer
from src.utils.config_parser import load_config, get_resolved_paths

logger = logging.getLogger("InspectAI.Dataset")


class CWRUVibrationDataset(Dataset):
    """PyTorch Dataset for 1D vibration windows with recording-level split filtering."""

    def __init__(
        self,
        npz_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        fold: Optional[int] = None,
        indices: Optional[Sequence[int]] = None,
        window_ids: Optional[Sequence[str]] = None,
        normalizer: Optional[VibrationNormalizer] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize PyTorch Dataset with optional partition filtering.

        Args:
            npz_path: Path to cwru_segmented_windows.npz. Defaults to data/processed/.
            metadata_path: Path to window_split_metadata.csv. Defaults to data/processed/.
            split: Partition to load: 'train', 'val', 'dev', 'test', or None (loads all).
            fold: Fold index (0 or 1) when split is 'train' or 'val'.
            indices: Optional subset of row indices to load.
            window_ids: Optional subset of window_ids to load.
            normalizer: Optional fitted VibrationNormalizer applied to windows.
            config: Optional project configuration dictionary.
        """
        super().__init__()
        self.config = config if config is not None else load_config()
        self.normalizer = normalizer
        resolved_paths = get_resolved_paths(self.config)

        # 1. Resolve NPZ file path
        if npz_path is None:
            self.npz_path = resolved_paths["processed_data_dir"] / "cwru_segmented_windows.npz"
        else:
            self.npz_path = Path(npz_path).resolve()

        if not self.npz_path.is_file():
            raise FileNotFoundError(f"Segmented windows archive not found: {self.npz_path}")

        # 2. Resolve Metadata CSV path (creates split metadata if not present)
        if metadata_path is None:
            self.metadata_path = resolved_paths["processed_data_dir"] / "window_split_metadata.csv"
        else:
            self.metadata_path = Path(metadata_path).resolve()

        if not self.metadata_path.is_file():
            logger.info("Split metadata not found. Generating via DatasetSplitter...")
            splitter = DatasetSplitter(config=self.config)
            self.metadata_path, _ = splitter.save_split_metadata(self.metadata_path)

        # 3. Load Arrays and Metadata
        npz_data = np.load(self.npz_path)
        self.raw_windows = npz_data["windows"]  # float32 array of shape (N_total, 2048)
        self.raw_labels = npz_data["labels"]    # int64 array of shape (N_total,)
        self.class_names = list(npz_data["class_names"])
        self.metadata_df = pd.read_csv(self.metadata_path)

        if len(self.raw_windows) != len(self.metadata_df):
            raise ValueError(
                f"Array length mismatch: {len(self.raw_windows)} windows in NPZ vs. "
                f"{len(self.metadata_df)} rows in metadata CSV."
            )

        # 4. Filter indices according to split / fold / subset arguments
        self.subset_indices = self._filter_indices(split=split, fold=fold, indices=indices, window_ids=window_ids)
        self.split_name = split
        self.fold = fold

        logger.info(
            f"Initialized CWRUVibrationDataset: {len(self.subset_indices)} samples "
            f"(split='{split}', fold={fold})."
        )

    def _filter_indices(
        self,
        split: Optional[str],
        fold: Optional[int],
        indices: Optional[Sequence[int]],
        window_ids: Optional[Sequence[str]],
    ) -> np.ndarray:
        """Filter global sample indices based on user request."""
        df = self.metadata_df

        if indices is not None:
            req_idx = np.asarray(indices, dtype=np.int64)
            if np.any(req_idx < 0) or np.any(req_idx >= len(df)):
                raise IndexError(f"Provided indices contain out-of-range values for dataset size {len(df)}.")
            return req_idx

        if window_ids is not None:
            id_set = set(window_ids)
            mask = df["window_id"].isin(id_set)
            return df[mask].index.values

        if split is None:
            return np.arange(len(df), dtype=np.int64)

        split_clean = split.lower().strip()

        if split_clean == "test":
            mask = df["split"] == "test"
            return df[mask].index.values

        if split_clean == "dev":
            mask = df["split"] == "dev"
            return df[mask].index.values

        if split_clean in ["train", "val"]:
            if fold is None:
                raise ValueError(f"When split='{split}', a fold index (e.g. fold=0 or fold=1) must be specified.")
            fold_col = f"fold_{fold}"
            if fold_col not in df.columns:
                raise ValueError(f"Fold column '{fold_col}' not found in metadata. Available: {[c for c in df.columns if c.startswith('fold_')]}")
            mask = df[fold_col] == split_clean
            return df[mask].index.values

        raise ValueError(f"Unrecognized split '{split}'. Expected 'train', 'val', 'dev', 'test', or None.")

    def __len__(self) -> int:
        """Return the number of vibration window samples in the subset."""
        return len(self.subset_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Retrieve a single vibration sample window, label, and metadata.

        Args:
            idx: Subset index (0 <= idx < len(self)).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
                - window_tensor: PyTorch float32 Tensor of shape (1, 2048) for 1D CNN
                - label_tensor: PyTorch long Tensor (scalar)
                - sample_meta: Dictionary of sample metadata attributes
        """
        if idx < 0 or idx >= len(self.subset_indices):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.subset_indices)}.")

        global_idx = self.subset_indices[idx]

        # Extract 1D window array (2048,) and reshape to (1, 2048) for PyTorch Conv1d
        window_arr = self.raw_windows[global_idx]
        window_tensor = torch.from_numpy(window_arr).float().unsqueeze(0)  # Shape (1, 2048)

        # Apply global training-only normalization if normalizer is provided
        if self.normalizer is not None:
            window_tensor = self.normalizer.transform_tensor(window_tensor)

        # Label tensor
        label_val = int(self.raw_labels[global_idx])
        label_tensor = torch.tensor(label_val, dtype=torch.long)

        # Metadata dictionary
        row = self.metadata_df.iloc[global_idx]
        sample_meta = {
            "window_id": row["window_id"],
            "source_file": row["source_file"],
            "recording_id": str(row["recording_id"]),
            "window_index": int(row["window_index"]),
            "fault_class": row["fault_class"],
            "class_id": label_val,
            "defect_diameter_in": float(row["defect_diameter_in"]),
            "motor_load_hp": int(row["motor_load_hp"]),
            "motor_rpm": float(row["motor_rpm"]) if pd.notna(row["motor_rpm"]) else float("nan"),
            "global_index": int(global_idx),
        }

        return window_tensor, label_tensor, sample_meta

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Convenience method to retrieve metadata without loading array tensor."""
        if idx < 0 or idx >= len(self.subset_indices):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self.subset_indices)}.")
        global_idx = self.subset_indices[idx]
        return self.metadata_df.iloc[global_idx].to_dict()


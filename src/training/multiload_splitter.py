"""Leakage-safe Leave-One-Load-Out (LOLO) dataset partitioning for multi-load CWRU benchmark.

This module implements the 3-fold Leave-One-Load-Out cross-validation strategy:
- Fold 1: Train on Loads 0, 2, 3 HP; Validate on Load 1 HP
- Fold 2: Train on Loads 0, 1, 3 HP; Validate on Load 2 HP
- Fold 3: Train on Loads 0, 1, 2 HP; Validate on Load 3 HP

Rules enforced:
1. Historical holdout recordings (97, 169, 185, 197) are permanently isolated from all folds.
2. No 0 HP validation fold is created (0 HP normal is in the protected holdout).
3. Zero window or recording leakage between training and validation splits.
4. Normalizer statistics are fitted exclusively on training windows in each fold.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.training.normalizer import VibrationNormalizer
from src.utils.config_parser import load_config, get_resolved_paths

logger = logging.getLogger("InspectAI.MultiLoadSplitter")

# Protected historical holdout recordings
PROTECTED_HOLDOUT_IDS: Set[str] = {"97", "169", "185", "197"}

# Expected SHA-256 digests for protected holdout files
EXPECTED_HOLDOUT_SHA256: Dict[str, str] = {
    "97": "16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c",
    "169": "f5448673725c4ea2735fe7d3fe4b16ad82b217eac6b62c4bf351c4549b0a5229",
    "185": "5bea2c60f177215ba3ec4ae287faf745d7e14aed1a12efe9e37b27c27044bafc",
    "197": "fa571b47cd942f9514a920b403fb7c053cd442bb2963a360a96d0f61432f03e9",
}


@dataclass
class LOLOFoldDefinition:
    """Specification of a single Leave-One-Load-Out fold."""

    fold_id: int
    val_load_hp: int
    train_loads_hp: List[int]
    train_recording_ids: List[str]
    val_recording_ids: List[str]
    train_window_count: int = 0
    val_window_count: int = 0
    train_class_counts: Dict[str, int] = field(default_factory=dict)
    val_class_counts: Dict[str, int] = field(default_factory=dict)
    train_severity_counts: Dict[str, int] = field(default_factory=dict)
    val_severity_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "val_load_hp": self.val_load_hp,
            "train_loads_hp": self.train_loads_hp,
            "train_recording_count": len(self.train_recording_ids),
            "val_recording_count": len(self.val_recording_ids),
            "train_recording_ids": self.train_recording_ids,
            "val_recording_ids": self.val_recording_ids,
            "train_window_count": self.train_window_count,
            "val_window_count": self.val_window_count,
            "train_class_counts": self.train_class_counts,
            "val_class_counts": self.val_class_counts,
            "train_severity_counts": self.train_severity_counts,
            "val_severity_counts": self.val_severity_counts,
        }


class MultiLoadSplitter:
    """Manages Leave-One-Load-Out (LOLO) dataset splitting across 40 CWRU recordings."""

    VALID_VAL_LOADS: List[int] = [1, 2, 3]

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        raw_data_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the multi-load splitter with manifest and validation checks."""
        self.config = config if config is not None else load_config()
        resolved_paths = get_resolved_paths(self.config)

        if manifest_path is None:
            self.manifest_path = Path("outputs/experiments/phase10_6/cwru_40_recordings_manifest.csv")
        else:
            self.manifest_path = Path(manifest_path).resolve()

        if raw_data_dir is None:
            self.raw_data_dir = resolved_paths["raw_data_dir"] / "cwru"
        else:
            self.raw_data_dir = Path(raw_data_dir).resolve()

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"40-recording manifest not found at: {self.manifest_path}")

        self.df_manifest = pd.read_csv(self.manifest_path)
        self.df_manifest["recording_id"] = self.df_manifest["recording_id"].astype(str)

        self._validate_manifest_structure()
        self._verify_holdout_integrity()
        self.folds = self._build_lolo_folds()

    def _validate_manifest_structure(self) -> None:
        """Verify that the manifest contains exactly 40 recordings and expected columns."""
        required_cols = [
            "recording_id", "filename", "fault_class", "fault_size_inches",
            "load_hp", "is_holdout", "sha256"
        ]
        missing_cols = [c for c in required_cols if c not in self.df_manifest.columns]
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {missing_cols}")

        if len(self.df_manifest) != 40:
            raise ValueError(f"Expected 40 recordings in manifest, found {len(self.df_manifest)}")

        # Verify holdout flags
        holdout_rows = self.df_manifest[self.df_manifest["recording_id"].isin(PROTECTED_HOLDOUT_IDS)]
        if len(holdout_rows) != 4:
            raise ValueError(f"Expected 4 holdout recordings in manifest, found {len(holdout_rows)}")

        if not (holdout_rows["is_holdout"] == True).all():
            raise ValueError("Protected holdout recordings must be explicitly flagged with is_holdout=True")

    def _verify_holdout_integrity(self) -> None:
        """Verify that raw holdout files exist and match their expected SHA-256 digests."""
        if not self.raw_data_dir.is_dir():
            logger.warning(f"Raw data dir {self.raw_data_dir} not accessible for hash verification.")
            return

        for rec_id, expected_sha in EXPECTED_HOLDOUT_SHA256.items():
            filepath = self.raw_data_dir / f"{rec_id}.mat"
            if not filepath.exists():
                raise FileNotFoundError(f"Protected holdout file missing: {filepath}")

            sha = hashlib.sha256()
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            actual_sha = sha.hexdigest()

            if actual_sha != expected_sha:
                raise ValueError(
                    f"SHA-256 mismatch for protected holdout {rec_id}.mat! "
                    f"Expected: {expected_sha}, Actual: {actual_sha}"
                )
        logger.info("Protected holdout integrity successfully verified (4/4 SHA-256 matched).")

    def _build_lolo_folds(self) -> Dict[int, LOLOFoldDefinition]:
        """Construct the 3 LOLO folds using development recordings only."""
        # Development recordings = non-holdout recordings (36 total)
        dev_df = self.df_manifest[~self.df_manifest["recording_id"].isin(PROTECTED_HOLDOUT_IDS)].copy()

        if len(dev_df) != 36:
            raise ValueError(f"Expected 36 development recordings, found {len(dev_df)}")

        folds: Dict[int, LOLOFoldDefinition] = {}

        for fold_idx, val_load in enumerate(self.VALID_VAL_LOADS, start=1):
            train_loads = [ld for ld in [0, 1, 2, 3] if ld != val_load]

            val_subset = dev_df[dev_df["load_hp"] == val_load]
            train_subset = dev_df[dev_df["load_hp"].isin(train_loads)]

            val_recs = sorted(val_subset["recording_id"].tolist())
            train_recs = sorted(train_subset["recording_id"].tolist())

            # Verification: zero overlap
            overlap = set(val_recs).intersection(set(train_recs))
            if overlap:
                raise ValueError(f"Leakage detected in Fold {fold_idx}: recordings {overlap} in both train and val!")

            # Verification: holdouts absent
            holdout_in_train = set(train_recs).intersection(PROTECTED_HOLDOUT_IDS)
            holdout_in_val = set(val_recs).intersection(PROTECTED_HOLDOUT_IDS)
            if holdout_in_train or holdout_in_val:
                raise ValueError(f"Holdout leakage in Fold {fold_idx}: Train={holdout_in_train}, Val={holdout_in_val}")

            fold_def = LOLOFoldDefinition(
                fold_id=fold_idx,
                val_load_hp=val_load,
                train_loads_hp=train_loads,
                train_recording_ids=train_recs,
                val_recording_ids=val_recs,
            )
            folds[fold_idx] = fold_def

        return folds

    def get_fold(self, fold_id: int) -> LOLOFoldDefinition:
        """Retrieve the specification for a given fold (1, 2, or 3)."""
        if fold_id not in self.folds:
            raise ValueError(f"Invalid fold_id {fold_id}. Must be one of {list(self.folds.keys())}")
        return self.folds[fold_id]

    def get_development_recordings(self) -> List[str]:
        """Return sorted list of all 36 development recording IDs."""
        dev_df = self.df_manifest[~self.df_manifest["recording_id"].isin(PROTECTED_HOLDOUT_IDS)]
        return sorted(dev_df["recording_id"].tolist())

    def get_holdout_recordings(self) -> List[str]:
        """Return sorted list of all 4 protected holdout recording IDs."""
        return sorted(list(PROTECTED_HOLDOUT_IDS))

    def fit_fold_normalizer(
        self,
        fold_id: int,
        training_windows: np.ndarray,
    ) -> VibrationNormalizer:
        """Fit a VibrationNormalizer strictly on training windows for a given fold.

        Args:
            fold_id: Fold index (1, 2, or 3).
            training_windows: 2D numpy array of shape (N_train, window_length).

        Returns:
            VibrationNormalizer: Fitted normalizer instance.
        """
        if fold_id not in self.folds:
            raise ValueError(f"Invalid fold_id {fold_id}")

        normalizer = VibrationNormalizer(strategy="global")
        normalizer.fit(training_windows)
        logger.info(
            f"Fitted normalizer for Fold {fold_id} on {len(training_windows)} windows: "
            f"mean={normalizer.mean:.6f}, std={normalizer.std:.6f}"
        )
        return normalizer

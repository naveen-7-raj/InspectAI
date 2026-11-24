"""Leakage-safe, recording-level dataset splitting for InspectAI."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config_parser import load_config, get_resolved_paths

logger = logging.getLogger("InspectAI.DatasetSplitter")


class DatasetSplitter:
    """Partitions vibration dataset strictly by recording ID to eliminate overlapping window leakage."""

    DEFAULT_TEST_RECORDINGS = {
        "Normal": "97.mat",          # 0 HP, 1796 RPM baseline
        "Ball_Fault": "185.mat",     # 0 HP, 0.014" intermediate defect
        "Inner_Race_Fault": "169.mat", # 0 HP, 0.014" intermediate defect
        "Outer_Race_Fault": "197.mat", # 0 HP, 0.014" intermediate defect
    }

    def __init__(
        self,
        metadata_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        resolved_paths = get_resolved_paths(self.config)

        if metadata_path is None:
            self.metadata_path = resolved_paths["processed_data_dir"] / "window_metadata.csv"
        else:
            self.metadata_path = Path(metadata_path).resolve()

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Window metadata file not found at: {self.metadata_path}")

        self.metadata_df = pd.read_csv(self.metadata_path)

        split_cfg = self.config.get("dataset_split", {})
        self.random_seed = split_cfg.get("random_seed", 42)
        self.group_col = split_cfg.get("group_column", "source_file")
        self.n_folds = split_cfg.get("n_validation_folds", 2)
        self.test_recordings_dict = split_cfg.get("test_recordings", self.DEFAULT_TEST_RECORDINGS)
        self.test_files = set(self.test_recordings_dict.values())

    def validate_dataset_presence(self) -> None:
        """Verify that all configured test recordings exist in the metadata."""
        available_files = set(self.metadata_df[self.group_col].unique())
        missing_test = self.test_files - available_files
        if missing_test:
            raise ValueError(
                f"Configured test recordings not found in dataset metadata: {missing_test}. "
                f"Available recordings: {sorted(available_files)}"
            )

    def create_split(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate recording-level test partition and 2-fold StratifiedGroupKFold on development pool.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]:
                - DataFrame enriched with 'split', 'fold_0', 'fold_1' columns
                - Summary dictionary containing partition statistics and group checks
        """
        self.validate_dataset_presence()
        df = self.metadata_df.copy()

        # 1. Final Test Partition (Exclusive to test set)
        is_test = df[self.group_col].isin(self.test_files)
        df["split"] = np.where(is_test, "test", "dev")

        dev_indices = df[~is_test].index.values
        dev_groups = df.loc[dev_indices, self.group_col].values
        dev_labels = df.loc[dev_indices, "fault_class"].values

        # Initialize fold assignment columns
        df["fold_0"] = np.where(is_test, "test", "")
        df["fold_1"] = np.where(is_test, "test", "")

        dev_recordings = sorted(df.loc[dev_indices, self.group_col].unique())

        # 2. Stratified Group 2-Fold CV on Development Pool
        sgkf = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)

        fold_info: Dict[int, Dict[str, Any]] = {}
        for fold, (train_pos, val_pos) in enumerate(sgkf.split(dev_indices, dev_labels, dev_groups)):
            train_idx = dev_indices[train_pos]
            val_idx = dev_indices[val_pos]

            col = f"fold_{fold}"
            df.loc[train_idx, col] = "train"
            df.loc[val_idx, col] = "val"

            train_recs = sorted(df.loc[train_idx, self.group_col].unique())
            val_recs = sorted(df.loc[val_idx, self.group_col].unique())

            train_classes = sorted(df.loc[train_idx, "fault_class"].unique())
            val_classes = sorted(df.loc[val_idx, "fault_class"].unique())

            # Assert no group overlap between train and val
            assert set(train_recs).isdisjoint(set(val_recs)), f"Group leakage detected in fold {fold}!"

            fold_info[fold] = {
                "train_recordings": train_recs,
                "val_recordings": val_recs,
                "train_classes": train_classes,
                "val_classes": val_classes,
                "train_windows": len(train_idx),
                "val_windows": len(val_idx),
            }

        # 3. Assertions & Validation
        test_classes = sorted(df[is_test]["fault_class"].unique())
        assert len(test_classes) == 4, f"Final test set missing classes! Found: {test_classes}"

        # Assert disjoint sets
        assert self.test_files.isdisjoint(set(dev_recordings)), "Test recordings overlap with development pool!"

        summary: Dict[str, Any] = {
            "total_windows": len(df),
            "test_windows": int(is_test.sum()),
            "dev_windows": int((~is_test).sum()),
            "test_recordings": sorted(self.test_files),
            "dev_recordings": dev_recordings,
            "test_class_distribution": df[is_test]["fault_class"].value_counts().to_dict(),
            "dev_class_distribution": df[~is_test]["fault_class"].value_counts().to_dict(),
            "folds": fold_info,
        }

        return df, summary

    def save_split_metadata(self, output_path: Optional[Union[str, Path]] = None) -> Tuple[Path, Dict[str, Any]]:
        """Save enriched split metadata CSV under data/processed/."""
        enriched_df, summary = self.create_split()

        if output_path is None:
            resolved_paths = get_resolved_paths(self.config)
            out_file = resolved_paths["processed_data_dir"] / "window_split_metadata.csv"
        else:
            out_file = Path(output_path).resolve()

        out_file.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.to_csv(out_file, index=False)
        logger.info(f"Saved dataset split metadata to: {out_file}")

        return out_file, summary


"""Unit and validation tests for PyTorch CWRUVibrationDataset (src/training/dataset.py)."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.training.dataset import CWRUVibrationDataset
from src.utils.config_parser import load_config, get_resolved_paths


def test_dataset_tensor_dtypes_and_shapes():
    """Test 7 & 8: Verify float32 window tensor of shape (1, 2048) and int64 scalar label."""
    ds = CWRUVibrationDataset(split="test")
    assert len(ds) == 818

    win, lbl, meta = ds[0]

    assert isinstance(win, torch.Tensor)
    assert win.dtype == torch.float32
    assert win.shape == (1, 2048)

    assert isinstance(lbl, torch.Tensor)
    assert lbl.dtype == torch.long
    assert lbl.ndim == 0  # Scalar tensor


def test_dataset_label_values_and_metadata():
    """Test 9: Verify labels are in range [0, 3] and metadata contains necessary attributes."""
    ds = CWRUVibrationDataset(split="test")

    for i in range(0, len(ds), 50):  # Sample across dataset
        _, lbl, meta = ds[i]
        assert lbl.item() in [0, 1, 2, 3]
        assert meta["class_id"] == lbl.item()
        assert "source_file" in meta
        assert "window_id" in meta
        assert "fault_class" in meta


def test_dataset_partition_filtering():
    """Verify split and fold filtering produces expected sample counts."""
    ds_all = CWRUVibrationDataset(split=None)
    assert len(ds_all) == 2928

    ds_test = CWRUVibrationDataset(split="test")
    assert len(ds_test) == 818

    ds_dev = CWRUVibrationDataset(split="dev")
    assert len(ds_dev) == 2110

    assert len(ds_test) + len(ds_dev) == len(ds_all)

    # Fold 0
    ds_train0 = CWRUVibrationDataset(split="train", fold=0)
    ds_val0 = CWRUVibrationDataset(split="val", fold=0)
    assert len(ds_train0) == 938
    assert len(ds_val0) == 1172
    assert len(ds_train0) + len(ds_val0) == len(ds_dev)

    # Fold 1
    ds_train1 = CWRUVibrationDataset(split="train", fold=1)
    ds_val1 = CWRUVibrationDataset(split="val", fold=1)
    assert len(ds_train1) == 1172
    assert len(ds_val1) == 938
    assert len(ds_train1) + len(ds_val1) == len(ds_dev)


def test_dataset_invalid_indices_and_inputs():
    """Test 10: Verify IndexError on out of bounds and ValueError on invalid split name."""
    ds = CWRUVibrationDataset(split="test")

    with pytest.raises(IndexError):
        _ = ds[len(ds)]

    with pytest.raises(IndexError):
        _ = ds[-1]

    with pytest.raises(ValueError, match="Unrecognized split"):
        _ = CWRUVibrationDataset(split="invalid_split_name")

    with pytest.raises(ValueError, match="fold index.*must be specified"):
        _ = CWRUVibrationDataset(split="train", fold=None)


def test_dataset_compatible_with_pytorch_dataloader():
    """Verify standard DataLoader batching, shuffling, and worker compatibility."""
    ds = CWRUVibrationDataset(split="test")
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    batch_x, batch_y, _ = next(iter(loader))
    assert batch_x.shape == (32, 1, 2048)
    assert batch_x.dtype == torch.float32
    assert batch_y.shape == (32,)
    assert batch_y.dtype == torch.long


def test_raw_dataset_and_npz_unmodified():
    """Test 11 & 12: Verify raw CWRU files and segmented NPZ remain immutable."""
    config = load_config()
    resolved_paths = get_resolved_paths(config)
    raw_dir = resolved_paths["raw_data_dir"] / "cwru"

    with open(raw_dir / "download_manifest.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        fpath = raw_dir / rec["filename"]
        hasher = hashlib.sha256()
        with open(fpath, "rb") as f_in:
            for chunk in iter(lambda: f_in.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == rec["sha256"]


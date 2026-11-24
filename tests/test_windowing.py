"""Unit and validation tests for WindowSegmenter (src/preprocessing/windowing.py)."""

import hashlib
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.windowing import WindowSegmenter, WindowMetadata, CWRU_CATALOG
from src.utils.config_parser import load_config, get_resolved_paths


@pytest.fixture
def segmenter():
    """Default WindowSegmenter configured with 12 kHz, 2048 window, 512 step."""
    return WindowSegmenter(window_size=2048, step_size=512, sampling_rate=12000)


def test_window_shape_and_dtype(segmenter):
    """Test 1: Verify each extracted window has exact shape (2048,) and float32 dtype."""
    dummy_signal = np.sin(np.linspace(0, 100, 10000))
    meta = {"file_name": "test.mat", "fault_class": "Normal", "class_id": 0}

    windows, meta_list = segmenter.segment_signal(dummy_signal, meta)
    assert windows.ndim == 2
    assert windows.shape[1] == 2048
    assert windows.dtype == np.float32
    assert len(windows) == len(meta_list)


def test_number_of_windows_formula(segmenter):
    """Test 2: Verify number of windows matches floor((L - W)/S) + 1 for various lengths."""
    test_lengths = [2048, 2049, 2559, 2560, 3072, 5000, 121265]
    w = 2048
    s = 512

    for length in test_lengths:
        expected = math.floor((length - w) / s) + 1
        computed = segmenter.compute_num_windows(length)
        assert computed == expected, f"Failed for length {length}: expected {expected}, got {computed}"

        # Test actual array segmentation
        sig = np.random.randn(length)
        windows, _ = segmenter.segment_signal(sig, {"file_name": "t.mat"})
        assert len(windows) == expected


def test_overlap_ratio_75_percent(segmenter):
    """Test 3: Verify consecutive windows overlap by exactly 75% (1536 samples)."""
    assert segmenter.overlap_ratio == 0.75
    overlap_samples = segmenter.window_size - segmenter.step_size
    assert overlap_samples == 1536

    # Verify identical values in overlapping region
    ramp = np.arange(10000, dtype=np.float64)
    windows, _ = segmenter.segment_signal(ramp, {"file_name": "ramp.mat"})

    # Window 0 tail (indices 512 to 2048) must match Window 1 head (indices 0 to 1536)
    assert np.array_equal(windows[0, 512:], windows[1, :1536])


def test_start_and_end_sample_indices(segmenter):
    """Test 4: Verify start and end sample indices are strictly sequential without gaps."""
    sig = np.arange(8000, dtype=np.float64)
    windows, meta_list = segmenter.segment_signal(sig, {"file_name": "sample_idx.mat"})

    for i, meta in enumerate(meta_list):
        expected_start = i * 512
        expected_end = expected_start + 2048

        assert meta.start_sample == expected_start
        assert meta.end_sample == expected_end
        assert np.array_equal(windows[i], sig[expected_start:expected_end])


def test_timestamps_calculation(segmenter):
    """Test 5: Verify timestamps in seconds match start/end sample divided by sampling rate."""
    sig = np.random.randn(6000)
    fs = segmenter.sampling_rate  # 12000
    _, meta_list = segmenter.segment_signal(sig, {"file_name": "ts.mat"})

    for meta in meta_list:
        expected_start_t = meta.start_sample / fs
        expected_end_t = meta.end_sample / fs
        assert np.isclose(meta.start_time_sec, expected_start_t, atol=1e-6)
        assert np.isclose(meta.end_time_sec, expected_end_t, atol=1e-6)
        assert np.isclose(meta.end_time_sec - meta.start_time_sec, 2048 / fs, atol=1e-6)


def test_metadata_preservation(segmenter):
    """Test 6: Verify fault class label, defect type, load and RPM are accurately assigned."""
    sig = np.random.randn(4000)
    meta_in = {
        "file_name": "105.mat",
        "fault_class": "Inner_Race_Fault",
        "class_id": 2,
        "defect_type": "Inner_Race",
        "defect_diameter_in": 0.007,
        "load_hp": 0,
        "rpm": 1797.0,
    }

    _, meta_list = segmenter.segment_signal(sig, meta_in)
    assert len(meta_list) > 0
    first = meta_list[0]

    assert first.source_file == "105.mat"
    assert first.recording_id == "105"
    assert first.fault_class == "Inner_Race_Fault"
    assert first.class_id == 2
    assert first.defect_type == "Inner_Race"
    assert first.defect_diameter_in == 0.007
    assert first.motor_load_hp == 0
    assert first.motor_rpm == 1797.0


def test_signal_shorter_than_window_raises_error(segmenter):
    """Test 7: Verify ValueError when signal is shorter than one complete window (no zero padding)."""
    short_signal = np.ones(2047)  # 1 sample short of 2048
    with pytest.raises(ValueError, match="is shorter than required window size"):
        segmenter.segment_signal(short_signal, {"file_name": "short.mat"})


def test_no_incomplete_final_windows(segmenter):
    """Test 8: Verify no partial or padded windows exist; every window has length 2048."""
    # Length that leaves a remainder of 300 samples
    remainder_len = 2048 + 3 * 512 + 300  # 3884 samples
    sig = np.random.randn(remainder_len)

    windows, meta_list = segmenter.segment_signal(sig, {"file_name": "rem.mat"})
    assert len(windows) == 4
    for w in windows:
        assert len(w) == 2048
        assert not np.isnan(w).any()


def test_no_windows_cross_recording_boundaries():
    """Test 9: Verify that segment_dataset processes files independently and windows never cross boundaries."""
    config = load_config()
    proc_dir = get_resolved_paths(config)["processed_data_dir"]
    meta_csv = proc_dir / "window_metadata.csv"

    assert meta_csv.is_file(), "window_metadata.csv does not exist"
    df = pd.read_csv(meta_csv)

    # Check each recording's windows: start_sample for window_index 0 must always be 0
    for file_name, group in df.groupby("source_file"):
        first_win = group[group["window_index"] == 0]
        assert len(first_win) == 1
        assert first_win["start_sample"].values[0] == 0


def test_invalid_input_handling(segmenter):
    """Test 10: Verify proper exception handling on empty, non-numeric, or NaN/Inf inputs."""
    # Empty
    with pytest.raises(ValueError, match="is empty"):
        segmenter.segment_signal(np.array([]), {"file_name": "empty.mat"})

    # Non-numeric
    with pytest.raises(TypeError, match="non-numeric"):
        segmenter.segment_signal(np.array(["a", "b"]), {"file_name": "str.mat"})

    # NaNs
    nan_sig = np.ones(3000)
    nan_sig[100] = np.nan
    with pytest.raises(ValueError, match="contains corrupt values"):
        segmenter.segment_signal(nan_sig, {"file_name": "nan.mat"})


def test_reproducible_segmentation(segmenter):
    """Test 11: Verify identical windows and metadata when segmenting the same signal repeatedly."""
    sig = np.sin(np.linspace(0, 50, 5000))
    meta = {"file_name": "repro.mat", "fault_class": "Normal", "class_id": 0}

    w1, m1 = segmenter.segment_signal(sig, meta)
    w2, m2 = segmenter.segment_signal(sig, meta)

    assert np.array_equal(w1, w2)
    assert len(m1) == len(m2)
    for a, b in zip(m1, m2):
        assert a == b


def test_raw_files_unmodified_after_segmentation():
    """Test 12: Verify that SHA-256 hashes of all 13 raw files match download_manifest.json."""
    config = load_config()
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"
    manifest_path = raw_dir / "download_manifest.json"

    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        fpath = raw_dir / rec["filename"]
        assert fpath.is_file()

        hasher = hashlib.sha256()
        with open(fpath, "rb") as f_in:
            for chunk in iter(lambda: f_in.read(65536), b""):
                hasher.update(chunk)

        assert hasher.hexdigest() == rec["sha256"], f"Integrity check failed for {rec['filename']}"


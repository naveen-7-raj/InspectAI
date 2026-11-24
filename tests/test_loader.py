"""Unit and validation tests for VibrationDataLoader (src/ingestion/loader.py)."""

from pathlib import Path
import numpy as np
import pytest
import scipy.io

from src.ingestion.loader import VibrationDataLoader, VibrationRecord
from src.utils.config_parser import load_config, get_resolved_paths


@pytest.fixture
def loader():
    config = load_config()
    raw_dir = get_resolved_paths(config)["raw_data_dir"] / "cwru"
    return VibrationDataLoader(raw_dir)


def test_loader_real_cwru_normal_files(loader):
    """Verify loading of authentic CWRU Normal baseline files."""
    record_97 = loader.load_file("97.mat")
    assert isinstance(record_97, VibrationRecord)
    assert record_97.file_name == "97.mat"
    assert "DE_time" in record_97.channels
    assert "FE_time" in record_97.channels
    assert "BA_time" in record_97.missing_channels
    assert record_97.sample_counts["DE_time"] == 243938
    assert record_97.channels["DE_time"].ndim == 1
    assert record_97.channels["DE_time"].dtype == np.float64
    assert record_97.rpm is not None
    assert int(record_97.rpm) in [1796, 1797]


def test_loader_real_cwru_collision_resolution(loader):
    """Verify that 99.mat disambiguates between X098 and X099 variables."""
    record_99 = loader.load_file("99.mat")
    assert record_99.raw_mat_keys["DE_time"] == "X099_DE_time"
    assert record_99.raw_mat_keys["FE_time"] == "X099_FE_time"
    assert record_99.sample_counts["DE_time"] == 485063


def test_loader_real_cwru_fault_files(loader):
    """Verify that fault files contain all three channels (DE, FE, BA)."""
    for fname in ["105.mat", "118.mat", "130.mat"]:
        record = loader.load_file(fname)
        assert "DE_time" in record.channels
        assert "FE_time" in record.channels
        assert "BA_time" in record.channels
        assert len(record.missing_channels) == 0
        assert record.sample_counts["DE_time"] > 120_000
        assert not np.isnan(record.channels["DE_time"]).any()
        assert not np.isinf(record.channels["DE_time"]).any()


def test_loader_missing_file_raises_error(loader):
    """Verify FileNotFoundError when target file does not exist."""
    with pytest.raises(FileNotFoundError):
        loader.load_file("non_existent_file_9999.mat")


def test_loader_missing_required_channel_raises_keyerror(tmp_path):
    """Verify KeyError when required primary channel DE_time is absent."""
    custom_mat = tmp_path / "missing_de.mat"
    scipy.io.savemat(custom_mat, {"FE_time": np.random.randn(1000, 1), "RPM": 1750})

    custom_loader = VibrationDataLoader(tmp_path)
    with pytest.raises(KeyError, match="Required primary channel 'DE_time' not found"):
        custom_loader.load_file("missing_de.mat")


def test_loader_nan_detection_raises_valueerror(tmp_path):
    """Verify ValueError when signal array contains NaN (no silent imputation)."""
    corrupt_data = np.array([1.0, 2.0, np.nan, 4.0])
    mat_path = tmp_path / "nan_corrupt.mat"
    scipy.io.savemat(mat_path, {"X101_DE_time": corrupt_data})

    custom_loader = VibrationDataLoader(tmp_path)
    with pytest.raises(ValueError, match="contains corrupt values: 1 NaNs"):
        custom_loader.load_file("nan_corrupt.mat")


def test_loader_inf_detection_raises_valueerror(tmp_path):
    """Verify ValueError when signal array contains Inf (no silent imputation)."""
    corrupt_data = np.array([1.0, 2.0, np.inf, 4.0])
    mat_path = tmp_path / "inf_corrupt.mat"
    scipy.io.savemat(mat_path, {"X101_DE_time": corrupt_data})

    custom_loader = VibrationDataLoader(tmp_path)
    with pytest.raises(ValueError, match="contains corrupt values.*1 Infs"):
        custom_loader.load_file("inf_corrupt.mat")


def test_loader_empty_array_raises_valueerror(tmp_path):
    """Verify ValueError when array has 0 samples."""
    mat_path = tmp_path / "empty_array.mat"
    scipy.io.savemat(mat_path, {"X101_DE_time": np.array([])})

    custom_loader = VibrationDataLoader(tmp_path)
    with pytest.raises(ValueError, match="is empty"):
        custom_loader.load_file("empty_array.mat")


def test_loader_multidimensional_matrix_raises_valueerror(tmp_path):
    """Verify ValueError when channel is a multidimensional image/matrix instead of vector."""
    mat_path = tmp_path / "multidim.mat"
    scipy.io.savemat(mat_path, {"X101_DE_time": np.ones((50, 50))})

    custom_loader = VibrationDataLoader(tmp_path)
    with pytest.raises(ValueError, match="is a 2D matrix.*not a single sensor vector"):
        custom_loader.load_file("multidim.mat")


def test_loader_corrupted_file_raises_valueerror(tmp_path):
    """Verify ValueError when .mat file contains garbage bytes."""
    corrupt_file = tmp_path / "garbage.mat"
    corrupt_file.write_bytes(b"This is not a valid matlab file header or payload.")

    custom_loader = VibrationDataLoader(tmp_path)
    with pytest.raises(ValueError, match="Failed to read MATLAB file"):
        custom_loader.load_file("garbage.mat")


def test_loader_different_variable_prefixes(tmp_path):
    """Verify dynamic key identification across various prefix styles."""
    test_cases = [
        ("no_prefix.mat", {"DE_time": np.ones(500), "FE_time": np.zeros(500)}),
        ("custom_prefix.mat", {"custom_sensor_DE_time": np.ones(500)}),
        ("lowercase.mat", {"x999_de_time": np.ones(500)}),
    ]

    custom_loader = VibrationDataLoader(tmp_path)
    for fname, data in test_cases:
        p = tmp_path / fname
        scipy.io.savemat(p, data)
        rec = custom_loader.load_file(fname)
        assert "DE_time" in rec.channels
        assert rec.sample_counts["DE_time"] == 500


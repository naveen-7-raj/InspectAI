"""External dataset loader module for cross-dataset validation.

Supports loading and extracting vibration time-series from independent benchmark datasets:
1. Paderborn University (PU) Bearing Dataset (MATLAB .mat structures).
2. MFPT Bearing Fault Dataset (MATLAB .mat bearing structures).
"""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.io as sio

logger = logging.getLogger("InspectAI.ExternalLoader")


@dataclass
class ExternalRecording:
    """Standardized container for loaded external vibration recordings."""

    recording_id: str
    dataset_name: str
    file_path: Path
    file_sha256: str
    signal: np.ndarray
    original_sampling_rate: float
    units: str
    fault_class: str  # "Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"
    bearing_model: str
    rotational_speed_rpm: float
    load_description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate recording attributes."""
        if not isinstance(self.signal, np.ndarray):
            self.signal = np.asarray(self.signal, dtype=np.float32)
        if self.signal.ndim != 1:
            self.signal = self.signal.flatten()
        if len(self.signal) == 0:
            raise ValueError(f"Empty signal in external recording: {self.recording_id}")


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA256 cryptographic hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ExternalDataLoader:
    """Loader for external bearing benchmarks (Paderborn University and MFPT)."""

    def __init__(self, external_dir: Optional[Union[str, Path]] = None):
        """Initialize external loader with optional base directory."""
        self.external_dir = Path(external_dir) if external_dir else Path("data/external")

    def load_paderborn_mat(
        self,
        mat_path: Union[str, Path],
        recording_id: Optional[str] = None,
        fault_class: str = "Unknown",
        bearing_model: str = "SKF 6203",
        rotational_speed_rpm: float = 1500.0,
        load_description: str = "1000 N radial, 0.7 Nm torque",
    ) -> ExternalRecording:
        """Parse Paderborn University KAt-DataCenter MATLAB file.

        Paderborn .mat files typically contain a struct with channel arrays:
        Channel 6/7 is vibration acceleration (m/s^2), sampled at 64,000 Hz.
        """
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"Paderborn file not found: {mat_path}")

        sha256_hash = compute_file_sha256(mat_path)
        mat_data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

        signal = None
        fs = 64000.0
        meta = {}

        # Paderborn structures often have root key matching the filename or 'data' / 'Y'
        candidates = [k for k in mat_data.keys() if not k.startswith("__")]
        
        for key in candidates:
            obj = mat_data[key]
            # Check for struct with Y channels (Paderborn format)
            if hasattr(obj, "Y"):
                y_channels = obj.Y
                if isinstance(y_channels, (list, np.ndarray)):
                    for ch in y_channels:
                        name = getattr(ch, "Name", "")
                        if "vibration" in str(name).lower() or "acceleration" in str(name).lower():
                            signal = np.asarray(getattr(ch, "Data", []), dtype=np.float32)
                            break
                    if signal is None and len(y_channels) > 6:
                        # Channel index 6 is standard vibration channel in Paderborn
                        ch6 = y_channels[6]
                        signal = np.asarray(getattr(ch6, "Data", []), dtype=np.float32)
            elif hasattr(obj, "vibration"):
                signal = np.asarray(obj.vibration, dtype=np.float32)
            elif isinstance(obj, np.ndarray) and obj.ndim == 1 and len(obj) > 1000:
                signal = obj.astype(np.float32)

        # Fallback to direct key check
        if signal is None:
            for k in candidates:
                val = mat_data[k]
                if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > 2000:
                    signal = val.astype(np.float32)
                    break

        if signal is None:
            raise ValueError(f"Could not extract vibration signal from Paderborn file: {mat_path}")

        rec_id = recording_id or f"EXT_PU_{mat_path.stem}"
        return ExternalRecording(
            recording_id=rec_id,
            dataset_name="Paderborn_University",
            file_path=mat_path,
            file_sha256=sha256_hash,
            signal=signal,
            original_sampling_rate=fs,
            units="m/s^2",
            fault_class=fault_class,
            bearing_model=bearing_model,
            rotational_speed_rpm=rotational_speed_rpm,
            load_description=load_description,
            metadata=meta,
        )

    def load_mfpt_mat(
        self,
        mat_path: Union[str, Path],
        recording_id: Optional[str] = None,
        fault_class: str = "Unknown",
        bearing_model: str = "Nice #1012",
        rotational_speed_rpm: float = 1000.0,
    ) -> ExternalRecording:
        """Parse MFPT MATLAB file.

        MFPT files typically contain a 'bearing' struct with:
        - gs: vibration acceleration array in g
        - sr: sampling rate (48828 or 97656)
        - load: radial load in lbs
        - rate: shaft rotational speed
        """
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"MFPT file not found: {mat_path}")

        sha256_hash = compute_file_sha256(mat_path)
        mat_data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

        signal = None
        fs = 48828.0
        load_lbs = 270.0
        rpm = rotational_speed_rpm

        if "bearing" in mat_data:
            b = mat_data["bearing"]
            if hasattr(b, "gs"):
                signal = np.asarray(b.gs, dtype=np.float32)
            if hasattr(b, "sr"):
                fs = float(b.sr)
            if hasattr(b, "load"):
                load_lbs = float(b.load)
            if hasattr(b, "rate"):
                rpm = float(b.rate) * 60.0 if float(b.rate) < 100 else float(b.rate)

        # Fallback to direct arrays
        if signal is None:
            for k in mat_data.keys():
                if not k.startswith("__"):
                    v = mat_data[k]
                    if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) > 2000:
                        signal = v.astype(np.float32)
                        break

        if signal is None:
            raise ValueError(f"Could not extract vibration signal from MFPT file: {mat_path}")

        rec_id = recording_id or f"EXT_MFPT_{mat_path.stem}"
        return ExternalRecording(
            recording_id=rec_id,
            dataset_name="MFPT",
            file_path=mat_path,
            file_sha256=sha256_hash,
            signal=signal,
            original_sampling_rate=fs,
            units="g",
            fault_class=fault_class,
            bearing_model=bearing_model,
            rotational_speed_rpm=rpm,
            load_description=f"{load_lbs} lbs radial load",
            metadata={"load_lbs": load_lbs},
        )

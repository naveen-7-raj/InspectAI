"""CWRU vibration streaming simulator for real-time edge monitoring simulation."""

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Union
import numpy as np

from src.ingestion.loader import VibrationDataLoader
from src.preprocessing.signal_cleaner import SignalCleaner
from src.preprocessing.windowing import CWRU_CATALOG
from src.utils.config_parser import get_resolved_paths, load_config

logger = logging.getLogger("InspectAI.StreamSimulator")

# 13 authentic CWRU benchmark recordings
ALLOWED_RECORDING_IDS = [
    "97",
    "98",
    "99",
    "100",
    "105",
    "118",
    "130",
    "169",
    "185",
    "197",
    "209",
    "222",
    "234",
]

ALLOWED_REPLAY_SPEEDS = [0.5, 1.0, 2.0, 5.0]


def normalize_recording_id(recording_id: Union[str, int]) -> str:
    """Normalize input recording identifier to clean string (e.g. '97.mat' -> '97')."""
    clean_id = str(recording_id).strip()
    if clean_id.endswith(".mat"):
        clean_id = clean_id[:-4]
    return clean_id


class CWRUStreamSimulator:
    """Replays authentic CWRU vibration recordings in chronological sliding windows.

    Simulates an industrial edge vibration sensor by:
    1. Loading an authentic CWRU recording (Drive-End accelerometer signal).
    2. Normalizing the sampling rate to 12 kHz with anti-aliasing and bandpass filtering.
    3. Generating sequential 2,048-sample windows stepping by 512 samples (75% overlap).
    4. Streaming windows with wall-clock pacing according to a configurable replay speed.
    """

    def __init__(
        self,
        recording_id: Union[str, int],
        raw_data_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        window_size: int = 2048,
        step_size: int = 512,
        target_sampling_rate: int = 12000,
    ) -> None:
        """Initialize CWRUStreamSimulator.

        Args:
            recording_id: Identifier of the CWRU recording (e.g. '97' or '97.mat').
            raw_data_dir: Directory containing raw .mat files. Defaults to data/raw/cwru.
            config: Optional project config dict.
            window_size: Number of samples per window (default: 2048).
            step_size: Number of samples to advance per window (default: 512).
            target_sampling_rate: Target sampling rate in Hz (default: 12000).

        Raises:
            ValueError: If recording_id is not in the 13 canonical CWRU recordings or parameters are invalid.
            FileNotFoundError: If the .mat file cannot be located.
        """
        self.recording_id = normalize_recording_id(recording_id)
        if self.recording_id not in ALLOWED_RECORDING_IDS:
            raise ValueError(
                f"Unknown CWRU recording ID '{recording_id}'. "
                f"Allowed recording IDs are: {', '.join(ALLOWED_RECORDING_IDS)}."
            )

        self.file_name = f"{self.recording_id}.mat"
        if self.file_name not in CWRU_CATALOG:
            raise ValueError(f"Recording '{self.file_name}' not found in CWRU catalog metadata.")

        self.catalog_entry = CWRU_CATALOG[self.file_name]
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.target_sampling_rate = int(target_sampling_rate)

        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}.")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}.")
        if self.step_size > self.window_size:
            raise ValueError(f"step_size ({self.step_size}) cannot exceed window_size ({self.window_size}).")

        # Resolve raw directory
        if raw_data_dir is not None:
            self.raw_data_dir = Path(raw_data_dir).resolve()
        else:
            cfg = config or load_config()
            paths = get_resolved_paths(cfg)
            self.raw_data_dir = (paths["raw_data_dir"] / "cwru").resolve()

        file_path = self.raw_data_dir / self.file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"CWRU raw recording file not found: {file_path}")

        # 1. Load raw MATLAB record
        loader = VibrationDataLoader(raw_data_dir=self.raw_data_dir)
        record = loader.load_file(self.file_name)

        if "DE_time" not in record.channels:
            raise KeyError(f"Recording '{self.file_name}' lacks required 'DE_time' accelerometer channel.")

        raw_de = record.channels["DE_time"]
        nominal_fs = int(self.catalog_entry.get("nominal_fs", 12000))

        # 2. Preprocess with SignalCleaner to target 12 kHz
        cfg = config or load_config()
        cleaner = SignalCleaner.from_config(cfg)
        cleaner.target_sampling_rate = self.target_sampling_rate

        processed = cleaner.clean_signal(
            signal=raw_de,
            orig_fs=nominal_fs,
            target_fs=self.target_sampling_rate,
        )
        self.signal_data = processed.signal.astype(np.float32)

        # 3. Calculate complete window count (no padding)
        if len(self.signal_data) < self.window_size:
            raise ValueError(
                f"Signal length ({len(self.signal_data)}) is shorter than window size ({self.window_size})."
            )

        self.total_windows = math_total_windows = (
            (len(self.signal_data) - self.window_size) // self.step_size
        ) + 1

        logger.info(
            f"Initialized CWRUStreamSimulator for recording {self.recording_id} "
            f"({self.catalog_entry['fault_class']}): {len(self.signal_data)} samples at {self.target_sampling_rate} Hz, "
            f"{self.total_windows} complete windows."
        )

    @property
    def fault_class(self) -> str:
        """Ground truth fault classification."""
        return self.catalog_entry["fault_class"]

    @property
    def class_id(self) -> int:
        """Ground truth class index."""
        return self.catalog_entry["class_id"]

    @property
    def defect_type(self) -> str:
        """Physical defect type description."""
        return self.catalog_entry["defect_type"]

    @property
    def load_hp(self) -> int:
        """Motor mechanical load in HP."""
        return self.catalog_entry["load_hp"]

    def get_window(self, index: int) -> Dict[str, Any]:
        """Extract a single window slice by sequential index.

        Args:
            index: Zero-based window index.

        Returns:
            Dict containing window metadata and raw float32 samples.
        """
        if index < 0 or index >= self.total_windows:
            raise IndexError(f"Window index {index} out of bounds [0, {self.total_windows - 1}].")

        start_sample = index * self.step_size
        end_sample = start_sample + self.window_size
        window_samples = self.signal_data[start_sample:end_sample]

        start_sec = round(start_sample / float(self.target_sampling_rate), 5)
        end_sec = round(end_sample / float(self.target_sampling_rate), 5)

        return {
            "sequence_number": index,
            "window_start_sample": start_sample,
            "window_end_sample": end_sample,
            "window_start_seconds": start_sec,
            "window_end_seconds": end_sec,
            "sampling_rate_hz": self.target_sampling_rate,
            "samples": window_samples,
            "recording_id": self.recording_id,
            "fault_class": self.fault_class,
            "class_id": self.class_id,
        }

    def iter_windows(self) -> Iterator[Dict[str, Any]]:
        """Synchronous generator yielding all windows in chronological order."""
        for i in range(self.total_windows):
            yield self.get_window(i)

    async def stream_windows(
        self,
        replay_speed: float = 1.0,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Asynchronous generator yielding windows with realistic wall-clock delay.

        Args:
            replay_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x faster, 0.5 = 2x slower).
            stop_event: Optional asyncio event to interrupt streaming early.

        Yields:
            Dict containing window metadata and samples.
        """
        if replay_speed <= 0:
            raise ValueError(f"Replay speed must be strictly positive, got {replay_speed}.")

        # Nominal time step between windows = step_size / sampling_rate
        # For step_size=512 and sampling_rate=12000, base delay = 512 / 12000 = 0.042667s (42.67 ms)
        base_interval_sec = self.step_size / float(self.target_sampling_rate)
        delay_sec = base_interval_sec / float(replay_speed)

        for i in range(self.total_windows):
            if stop_event is not None and stop_event.is_set():
                logger.info(f"Stream interrupted by stop event at window {i}/{self.total_windows}.")
                break

            yield self.get_window(i)

            # Avoid sleep after the last window
            if i < self.total_windows - 1 and delay_sec > 0:
                await asyncio.sleep(delay_sec)


class VibrationStreamSimulator:
    """Backwards-compatible streaming simulator wrapper accepting in-memory numpy arrays."""

    def __init__(
        self,
        signal_data: np.ndarray,
        sampling_rate: int = 12000,
        chunk_size: int = 512,
        window_size: int = 2048,
    ) -> None:
        self.signal_data = np.asarray(signal_data, dtype=np.float32).reshape(-1)
        self.sampling_rate = int(sampling_rate)
        self.chunk_size = int(chunk_size)
        self.window_size = int(window_size)

        if len(self.signal_data) >= self.window_size:
            self.total_windows = ((len(self.signal_data) - self.window_size) // self.chunk_size) + 1
        else:
            self.total_windows = 0

    def stream_chunks(self) -> Iterator[np.ndarray]:
        """Yield consecutive 2,048-sample windows sliding by chunk_size."""
        for i in range(self.total_windows):
            start = i * self.chunk_size
            end = start + self.window_size
            yield self.signal_data[start:end]

    async def async_stream_chunks(
        self,
        delay_seconds: float = 0.04267,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield consecutive windows with asynchronous sleep delay."""
        for i in range(self.total_windows):
            start = i * self.chunk_size
            end = start + self.window_size
            yield self.signal_data[start:end]
            if i < self.total_windows - 1 and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

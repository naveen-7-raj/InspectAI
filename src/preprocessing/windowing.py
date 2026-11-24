"""Sliding window segmentation engine for vibration time series."""

from dataclasses import dataclass, asdict
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.ingestion.loader import VibrationDataLoader, VibrationRecord
from src.preprocessing.signal_cleaner import SignalCleaner, ProcessedSignal
from src.utils.config_parser import load_config, get_resolved_paths

logger = logging.getLogger("InspectAI.Windowing")

# Ground truth metadata mapping for the 13 CWRU recordings
CWRU_CATALOG: Dict[str, Dict[str, Any]] = {
    "97.mat": {"fault_class": "Normal", "class_id": 0, "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 0, "nominal_fs": 48000},
    "98.mat": {"fault_class": "Normal", "class_id": 0, "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 1, "nominal_fs": 48000},
    "99.mat": {"fault_class": "Normal", "class_id": 0, "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 2, "nominal_fs": 48000},
    "100.mat": {"fault_class": "Normal", "class_id": 0, "defect_type": "None", "defect_diameter_in": 0.0, "load_hp": 3, "nominal_fs": 48000},
    "118.mat": {"fault_class": "Ball_Fault", "class_id": 1, "defect_type": "Ball", "defect_diameter_in": 0.007, "load_hp": 0, "nominal_fs": 12000},
    "185.mat": {"fault_class": "Ball_Fault", "class_id": 1, "defect_type": "Ball", "defect_diameter_in": 0.014, "load_hp": 0, "nominal_fs": 12000},
    "222.mat": {"fault_class": "Ball_Fault", "class_id": 1, "defect_type": "Ball", "defect_diameter_in": 0.021, "load_hp": 0, "nominal_fs": 12000},
    "105.mat": {"fault_class": "Inner_Race_Fault", "class_id": 2, "defect_type": "Inner_Race", "defect_diameter_in": 0.007, "load_hp": 0, "nominal_fs": 12000},
    "169.mat": {"fault_class": "Inner_Race_Fault", "class_id": 2, "defect_type": "Inner_Race", "defect_diameter_in": 0.014, "load_hp": 0, "nominal_fs": 12000},
    "209.mat": {"fault_class": "Inner_Race_Fault", "class_id": 2, "defect_type": "Inner_Race", "defect_diameter_in": 0.021, "load_hp": 0, "nominal_fs": 12000},
    "130.mat": {"fault_class": "Outer_Race_Fault", "class_id": 3, "defect_type": "Outer_Race_6_00", "defect_diameter_in": 0.007, "load_hp": 0, "nominal_fs": 12000},
    "197.mat": {"fault_class": "Outer_Race_Fault", "class_id": 3, "defect_type": "Outer_Race_6_00", "defect_diameter_in": 0.014, "load_hp": 0, "nominal_fs": 12000},
    "234.mat": {"fault_class": "Outer_Race_Fault", "class_id": 3, "defect_type": "Outer_Race_6_00", "defect_diameter_in": 0.021, "load_hp": 0, "nominal_fs": 12000},
}


@dataclass
class WindowMetadata:
    """Metadata describing a single extracted vibration window."""

    window_id: str
    source_file: str
    recording_id: str
    window_index: int
    start_sample: int
    end_sample: int
    start_time_sec: float
    end_time_sec: float
    sampling_rate_hz: int
    fault_class: str
    class_id: int
    defect_type: str
    defect_diameter_in: float
    motor_load_hp: int
    motor_rpm: Optional[float]


class WindowSegmenter:
    """Segments continuous preprocessed vibration signals into fixed-length overlapping windows."""

    def __init__(
        self,
        window_size: int = 2048,
        step_size: int = 512,
        sampling_rate: int = 12000,
        drop_incomplete: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        if step_size > window_size:
            raise ValueError(f"step_size ({step_size}) cannot exceed window_size ({window_size}).")
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}.")

        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.sampling_rate = int(sampling_rate)
        self.drop_incomplete = bool(drop_incomplete)
        self.overlap_ratio = (self.window_size - self.step_size) / self.window_size

    def compute_num_windows(self, signal_length: int) -> int:
        """Calculate exact number of complete non-padded windows for a given signal length.

        Formula: floor((signal_length - window_size) / step_size) + 1
        """
        if signal_length < self.window_size:
            return 0
        return math.floor((signal_length - self.window_size) / self.step_size) + 1

    def segment_signal(
        self,
        signal: np.ndarray,
        source_metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[WindowMetadata]]:
        """Slice a continuous 1D vibration signal into overlapping windows.

        Args:
            signal: 1D vibration array.
            source_metadata: Metadata describing source file, fault class, rpm, load.

        Returns:
            Tuple[np.ndarray, List[WindowMetadata]]:
                - Windows array of shape (num_windows, window_size)
                - List of WindowMetadata records corresponding to each window

        Raises:
            ValueError: If signal is shorter than window_size or contains invalid values.
        """
        valid_signal = SignalCleaner.validate_signal(signal, name="segment_input")
        sig_len = len(valid_signal)

        if sig_len < self.window_size:
            raise ValueError(
                f"Signal length ({sig_len} samples) is shorter than required window size ({self.window_size} samples). "
                f"Padding incomplete windows is prohibited."
            )

        num_windows = self.compute_num_windows(sig_len)
        windows = np.empty((num_windows, self.window_size), dtype=np.float32)
        metadata_list: List[WindowMetadata] = []

        source_file = source_metadata.get("file_name", "unknown.mat")
        rec_id = Path(source_file).stem
        fault_class = source_metadata.get("fault_class", "Unknown")
        class_id = int(source_metadata.get("class_id", -1))
        defect_type = source_metadata.get("defect_type", "Unknown")
        defect_diam = float(source_metadata.get("defect_diameter_in", 0.0))
        load_hp = int(source_metadata.get("load_hp", 0))
        rpm_val = source_metadata.get("rpm", None)

        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            windows[i] = valid_signal[start_idx:end_idx].astype(np.float32)

            start_t = start_idx / self.sampling_rate
            end_t = end_idx / self.sampling_rate

            win_meta = WindowMetadata(
                window_id=f"{rec_id}_w{i:04d}",
                source_file=source_file,
                recording_id=rec_id,
                window_index=i,
                start_sample=start_idx,
                end_sample=end_idx,
                start_time_sec=round(start_t, 6),
                end_time_sec=round(end_t, 6),
                sampling_rate_hz=self.sampling_rate,
                fault_class=fault_class,
                class_id=class_id,
                defect_type=defect_type,
                defect_diameter_in=defect_diam,
                motor_load_hp=load_hp,
                motor_rpm=rpm_val,
            )
            metadata_list.append(win_meta)

        logger.debug(
            f"Segmented {source_file}: {sig_len} samples -> {num_windows} windows of shape ({self.window_size},)."
        )
        return windows, metadata_list

    @classmethod
    def segment_dataset(
        cls,
        config: Optional[Dict[str, Any]] = None,
        save_dataset: bool = True,
    ) -> Dict[str, Any]:
        """Process all 13 authentic CWRU recordings one at a time and export segmented dataset.

        Args:
            config: Optional loaded configuration dictionary.
            save_dataset: Whether to save numerical arrays and metadata CSV to data/processed/.

        Returns:
            Dict[str, Any]: Execution summary, class counts, and file references.
        """
        if config is None:
            config = load_config()

        resolved_paths = get_resolved_paths(config)
        raw_dir = resolved_paths["raw_data_dir"] / "cwru"
        proc_dir = resolved_paths["processed_data_dir"]
        proc_dir.mkdir(parents=True, exist_ok=True)

        win_cfg = config.get("windowing", {})
        win_size = win_cfg.get("window_size", 2048)
        step_sz = win_cfg.get("step_size", 512)
        fs = win_cfg.get("sampling_rate_hz", 12000)

        segmenter = cls(window_size=win_size, step_size=step_sz, sampling_rate=fs)

        prep_cfg = config["preprocessing"]
        cleaner = SignalCleaner(
            target_sampling_rate=fs,
            resample_method=prep_cfg["resample_method"],
            remove_dc=prep_cfg["remove_dc"],
            filter_enabled=prep_cfg["filtering"]["enabled"],
            filter_type=prep_cfg["filtering"]["type"],
            low_cutoff_hz=prep_cfg["filtering"]["low_cutoff_hz"],
            high_cutoff_hz=prep_cfg["filtering"]["high_cutoff_hz"],
            filter_order=prep_cfg["filtering"]["order"],
        )

        loader = VibrationDataLoader(raw_dir)

        all_windows_list: List[np.ndarray] = []
        all_metadata_list: List[Dict[str, Any]] = []
        recording_summaries: List[Dict[str, Any]] = []

        logger.info(f"Starting dataset window segmentation (window={win_size}, step={step_sz}, overlap=75%)...")

        # Process one recording at a time to keep RAM minimal
        for fname, meta in CWRU_CATALOG.items():
            rec = loader.load_file(fname)
            raw_de = rec.channels["DE_time"]

            # Preprocess to 12 kHz, remove DC, apply filter
            processed = cleaner.clean_signal(raw_de, orig_fs=meta["nominal_fs"])

            merged_meta = dict(meta)
            merged_meta["file_name"] = fname
            merged_meta["rpm"] = rec.rpm

            # Segment into windows
            windows, meta_list = segmenter.segment_signal(processed.signal, merged_meta)

            all_windows_list.append(windows)
            all_metadata_list.extend([asdict(m) for m in meta_list])

            recording_summaries.append(
                {
                    "file_name": fname,
                    "category": meta["fault_class"],
                    "cleaned_samples": len(processed.signal),
                    "num_windows": len(windows),
                    "resampled": processed.applied_resampling,
                }
            )

        # Concatenate windows into a single unified array
        all_windows = np.concatenate(all_windows_list, axis=0)  # Shape (N_total, 2048)
        all_labels = np.array([m["class_id"] for m in all_metadata_list], dtype=np.int64)

        metadata_df = pd.DataFrame(all_metadata_list)

        results: Dict[str, Any] = {
            "total_windows": len(all_windows),
            "window_shape": list(all_windows.shape),
            "sampling_rate_hz": fs,
            "classes": config["dataset"]["classes"],
            "class_counts": metadata_df["fault_class"].value_counts().to_dict(),
            "recording_summaries": recording_summaries,
            "npz_path": None,
            "csv_path": None,
        }

        if save_dataset:
            # 1. Save compressed numerical array (ideal for PyTorch DataLoader)
            npz_out = proc_dir / "cwru_segmented_windows.npz"
            np.savez_compressed(
                npz_out,
                windows=all_windows,
                labels=all_labels,
                class_names=np.array(config["dataset"]["classes"]),
            )
            results["npz_path"] = str(npz_out)
            logger.info(f"Saved numerical windows array to: {npz_out} ({npz_out.stat().st_size / 1e6:.2f} MB)")

            # 2. Save lightweight window metadata index CSV
            csv_out = proc_dir / "window_metadata.csv"
            metadata_df.to_csv(csv_out, index=False)
            results["csv_path"] = str(csv_out)
            logger.info(f"Saved window metadata catalog to: {csv_out}")

        return results


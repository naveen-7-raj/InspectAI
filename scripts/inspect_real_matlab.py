"""Inspection script for actual Paderborn University MATLAB (.mat) files.

Validates the internal structure, vibration channel index, sample counts,
sampling rates, and physical signal characteristics for pilot bearings:
- K001 (Healthy)
- KA04 (Outer Race Fault)
- KI04 (Inner Race Fault)
- KB23 (Rolling Element / Ball Fault)
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXTRACTED_DIR = REPO_ROOT / "data" / "external" / "extracted"
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments" / "phase14"


def inspect_mat_file(mat_path: Path) -> Dict[str, Any]:
    """Inspect a single Paderborn MATLAB file and return structured metadata."""
    data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    candidates = [k for k in data.keys() if not k.startswith("__")]
    if not candidates:
        raise ValueError(f"No valid root key found in {mat_path.name}")
    
    root_key = candidates[0]
    root_obj = data[root_key]

    fields = [attr for attr in dir(root_obj) if not attr.startswith("_")]
    
    # Inspect channels in Y
    y_channels = getattr(root_obj, "Y", [])
    channel_info = []
    vib_data = None
    vib_channel_idx = -1

    for idx, ch in enumerate(y_channels):
        name = getattr(ch, "Name", f"Channel_{idx}")
        unit = getattr(ch, "Unit", "Unknown")
        ch_data = getattr(ch, "Data", np.array([]))
        
        info = {
            "channel_index": idx,
            "name": str(name),
            "unit": str(unit),
            "length": len(ch_data) if hasattr(ch_data, "__len__") else 0,
        }
        channel_info.append(info)
        
        if "vibration" in str(name).lower() or "acceleration" in str(name).lower() or idx == 6:
            vib_data = np.asarray(ch_data, dtype=np.float32)
            vib_channel_idx = idx

    # Inspect sampling frequency from X
    fs_nominal = 64000.0
    x_obj = getattr(root_obj, "X", None)
    if x_obj is not None and hasattr(x_obj, "Raster"):
        # Raster is delta-t in seconds (e.g. 1.5625e-05 s -> 64000 Hz)
        raster = float(getattr(x_obj, "Raster", 1.0 / 64000.0))
        if raster > 0:
            fs_nominal = round(1.0 / raster, 2)

    # Compute signal metrics if vibration data found
    signal_metrics = {}
    if vib_data is not None and len(vib_data) > 0:
        rms = float(np.sqrt(np.mean(vib_data ** 2)))
        peak = float(np.max(np.abs(vib_data)))
        crest = float(peak / (rms + 1e-8))
        kurt = float(np.mean(((vib_data - np.mean(vib_data)) / (np.std(vib_data) + 1e-8)) ** 4))
        signal_metrics = {
            "num_samples": len(vib_data),
            "duration_seconds": float(len(vib_data) / fs_nominal),
            "min_val": float(np.min(vib_data)),
            "max_val": float(np.max(vib_data)),
            "mean_val": float(np.mean(vib_data)),
            "std_val": float(np.std(vib_data)),
            "rms_val": rms,
            "peak_val": peak,
            "crest_factor": crest,
            "kurtosis": kurt,
        }

    return {
        "filename": mat_path.name,
        "root_key": root_key,
        "root_fields": fields,
        "sampling_rate_hz": fs_nominal,
        "vibration_channel_index": vib_channel_idx,
        "channels": channel_info,
        "signal_metrics": signal_metrics,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bearings = ["K001", "KA04", "KI04", "KB23"]
    inspection_results = {}

    for b_id in bearings:
        b_dir = EXTRACTED_DIR / b_id
        if not b_dir.exists():
            print(f"Directory not found for {b_id}: {b_dir}")
            continue
        
        mat_files = sorted(b_dir.rglob("*.mat"))
        print(f"Found {len(mat_files)} .mat files for {b_id}")
        if not mat_files:
            continue
        
        # Pick 3 representative files: e.g. N15_M07_F10_*, N15_M01_F10_*, N09_M07_F10_*
        representative_files = []
        for condition in ["N15_M07_F10", "N15_M01_F10", "N09_M07_F10"]:
            matched = [f for f in mat_files if condition in f.name]
            if matched:
                representative_files.append(matched[0])
            elif mat_files:
                representative_files.append(mat_files[0])
        
        # Deduplicate
        seen = set()
        unique_rep = []
        for f in representative_files:
            if f.name not in seen:
                seen.add(f.name)
                unique_rep.append(f)

        inspections = []
        for f in unique_rep:
            print(f"  Inspecting {f.name}...")
            meta = inspect_mat_file(f)
            inspections.append(meta)

        inspection_results[b_id] = {
            "bearing_id": b_id,
            "total_files_found": len(mat_files),
            "sample_inspections": inspections,
        }

    out_file = OUTPUT_DIR / "real_matlab_inspection.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(inspection_results, f, indent=2)
    print(f"Saved inspection summary to {out_file}")


if __name__ == "__main__":
    main()


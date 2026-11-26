"""Acquisition script for Phase 10.6 CWRU Multi-Load Dataset Expansion.

Downloads 27 additional 12 kHz Drive-End (DE) fault recordings at 1 HP, 2 HP, and 3 HP
from the official Case Western Reserve University Bearing Data Center.
Expands the raw dataset from 13 to 40 authentic recordings.

Preserves historical holdout recordings (97, 169, 185, 197) completely intact.
"""

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List
import urllib.request
import scipy.io as sio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.MultiLoadAcquisition")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CWRU_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "phase10_6"

OFFICIAL_BASE_URL = "https://engineering.case.edu/sites/default/files"

# 27 New Recordings Specification
NEW_RECORDINGS_SPEC = [
    # Inner Race 0.007"
    {"id": "106", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.007, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "107", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.007, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "108", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.007, "load_hp": 3, "nominal_rpm": 1730},
    # Inner Race 0.014"
    {"id": "170", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.014, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "171", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.014, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "172", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.014, "load_hp": 3, "nominal_rpm": 1730},
    # Inner Race 0.021"
    {"id": "210", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.021, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "211", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.021, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "212", "fault_class": "Inner_Race_Fault", "defect_diameter_in": 0.021, "load_hp": 3, "nominal_rpm": 1730},

    # Ball Fault 0.007"
    {"id": "119", "fault_class": "Ball_Fault", "defect_diameter_in": 0.007, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "120", "fault_class": "Ball_Fault", "defect_diameter_in": 0.007, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "121", "fault_class": "Ball_Fault", "defect_diameter_in": 0.007, "load_hp": 3, "nominal_rpm": 1730},
    # Ball Fault 0.014"
    {"id": "186", "fault_class": "Ball_Fault", "defect_diameter_in": 0.014, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "187", "fault_class": "Ball_Fault", "defect_diameter_in": 0.014, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "188", "fault_class": "Ball_Fault", "defect_diameter_in": 0.014, "load_hp": 3, "nominal_rpm": 1730},
    # Ball Fault 0.021"
    {"id": "223", "fault_class": "Ball_Fault", "defect_diameter_in": 0.021, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "224", "fault_class": "Ball_Fault", "defect_diameter_in": 0.021, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "225", "fault_class": "Ball_Fault", "defect_diameter_in": 0.021, "load_hp": 3, "nominal_rpm": 1730},

    # Outer Race 0.007"
    {"id": "131", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.007, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "132", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.007, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "133", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.007, "load_hp": 3, "nominal_rpm": 1730},
    # Outer Race 0.014"
    {"id": "198", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.014, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "199", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.014, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "200", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.014, "load_hp": 3, "nominal_rpm": 1730},
    # Outer Race 0.021"
    {"id": "235", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.021, "load_hp": 1, "nominal_rpm": 1772},
    {"id": "236", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.021, "load_hp": 2, "nominal_rpm": 1750},
    {"id": "237", "fault_class": "Outer_Race_Fault", "defect_diameter_in": 0.021, "load_hp": 3, "nominal_rpm": 1730},
]

# Historical Holdout IDs that MUST remain protected and excluded from modification
HISTORICAL_HOLDOUT_IDS = {"97", "169", "185", "197"}


def compute_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 digest."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_and_verify_recordings() -> Dict[str, Any]:
    """Download and verify all 27 additional recordings from CWRU."""
    RAW_CWRU_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    results = []
    success_count = 0
    failed_count = 0

    logger.info(f"Initiating acquisition of {len(NEW_RECORDINGS_SPEC)} CWRU multi-load fault recordings...")

    for spec in NEW_RECORDINGS_SPEC:
        rec_id = spec["id"]
        filename = f"{rec_id}.mat"
        target_path = RAW_CWRU_DIR / filename
        url = f"{OFFICIAL_BASE_URL}/{filename}"

        if rec_id in HISTORICAL_HOLDOUT_IDS:
            raise RuntimeError(f"FATAL: Attempted to download or overwrite holdout ID: {rec_id}")

        rec_entry = {
            "recording_id": rec_id,
            "filename": filename,
            "fault_class": spec["fault_class"],
            "defect_diameter_in": spec["defect_diameter_in"],
            "load_hp": spec["load_hp"],
            "nominal_rpm": spec["nominal_rpm"],
            "url": url,
            "status": "PENDING",
            "sha256": None,
            "file_size_bytes": 0,
            "is_valid_matlab": False,
            "de_channel": None,
            "signal_length": 0,
            "verified_rpm": None,
            "rpm_status": "Missing",
            "channels": [],
            "error": None,
        }

        # 1. Download if not present
        if not target_path.exists() or target_path.stat().st_size == 0:
            logger.info(f"Downloading {filename} from {url}...")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                    if len(data) < 5000:
                        raise ValueError(f"Payload too small: {len(data)} bytes")
                    with open(target_path, "wb") as f_out:
                        f_out.write(data)
                logger.info(f"Successfully downloaded {filename} ({len(data):,} bytes)")
                time.sleep(0.3)  # Polite pacing
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                rec_entry["status"] = "DOWNLOAD_FAILED"
                rec_entry["error"] = str(e)
                failed_count += 1
                results.append(rec_entry)
                continue
        else:
            logger.info(f"File {filename} already exists locally ({target_path.stat().st_size:,} bytes). Verifying...")

        # 2. Inspect File Properties & SHA-256
        rec_entry["file_size_bytes"] = target_path.stat().st_size
        rec_entry["sha256"] = compute_sha256(target_path)

        # 3. Inspect MATLAB Structure dynamically
        try:
            mat_dict = sio.loadmat(str(target_path))
            rec_entry["is_valid_matlab"] = True
            keys = [k for k in mat_dict.keys() if not k.startswith("__")]
            rec_entry["channels"] = keys

            # Identify DE channel dynamically (e.g. X106_DE_time, X106DE_time, etc.)
            de_key = None
            for k in keys:
                if "DE" in k and "time" in k:
                    de_key = k
                    break

            if de_key is None:
                raise ValueError(f"No Drive End (DE) time channel found in keys: {keys}")

            de_arr = mat_dict[de_key]
            rec_entry["de_channel"] = de_key
            rec_entry["signal_length"] = int(de_arr.size)

            # Check for RPM variable
            rpm_key = next((k for k in keys if "RPM" in k.upper()), None)
            if rpm_key is not None:
                try:
                    val = float(mat_dict[rpm_key].squeeze())
                    rec_entry["verified_rpm"] = val
                    rec_entry["rpm_status"] = "Verified (MATLAB Header)"
                except Exception:
                    rec_entry["rpm_status"] = "Present but non-scalar"
            else:
                rec_entry["rpm_status"] = f"Missing in Header (Nominal: {spec['nominal_rpm']})"

            rec_entry["status"] = "SUCCESS"
            success_count += 1
            logger.info(f"Verified {filename}: DE='{de_key}', length={rec_entry['signal_length']}, RPM={rec_entry['verified_rpm']}")

        except Exception as e:
            logger.error(f"Validation error in {filename}: {e}")
            rec_entry["status"] = "VALIDATION_FAILED"
            rec_entry["error"] = str(e)
            failed_count += 1

        results.append(rec_entry)

    summary = {
        "total_requested": len(NEW_RECORDINGS_SPEC),
        "success_count": success_count,
        "failed_count": failed_count,
        "holdout_protection_verified": True,
        "records": results,
    }

    # Save verification JSON
    with open(OUTPUT_DIR / "download_verification.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Acquisition finished: {success_count}/{len(NEW_RECORDINGS_SPEC)} successful. Summary saved to download_verification.json")
    return summary


if __name__ == "__main__":
    download_and_verify_recordings()


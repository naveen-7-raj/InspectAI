"""InspectAI - CWRU Bearing Dataset Acquisition & Verification Pipeline.

Downloads genuine industrial vibration recordings from the Case Western Reserve
University Bearing Data Center, verifies MATLAB structure, computes SHA-256 checksums,
and generates structured manifests.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import urllib.request
import urllib.error
import scipy.io

OFFICIAL_BASE_URL = "https://engineering.case.edu/sites/default/files"

DATASET_SPEC = {
    "NORMAL": [
        {"filename": "97.mat", "fault_type": "Normal", "load_hp": 0, "rpm": 1797, "diameter_in": 0.0},
        {"filename": "98.mat", "fault_type": "Normal", "load_hp": 1, "rpm": 1772, "diameter_in": 0.0},
        {"filename": "99.mat", "fault_type": "Normal", "load_hp": 2, "rpm": 1750, "diameter_in": 0.0},
        {"filename": "100.mat", "fault_type": "Normal", "load_hp": 3, "rpm": 1730, "diameter_in": 0.0},
    ],
    "BALL_FAULT": [
        {"filename": "118.mat", "fault_type": "Ball_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.007},
        {"filename": "185.mat", "fault_type": "Ball_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.014},
        {"filename": "222.mat", "fault_type": "Ball_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.021},
    ],
    "INNER_RACE_FAULT": [
        {"filename": "105.mat", "fault_type": "Inner_Race_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.007},
        {"filename": "169.mat", "fault_type": "Inner_Race_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.014},
        {"filename": "209.mat", "fault_type": "Inner_Race_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.021},
    ],
    "OUTER_RACE_FAULT": [
        {"filename": "130.mat", "fault_type": "Outer_Race_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.007},
        {"filename": "197.mat", "fault_type": "Outer_Race_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.014},
        {"filename": "234.mat", "fault_type": "Outer_Race_Fault", "load_hp": 0, "rpm": 1797, "diameter_in": 0.021},
    ],
}


def compute_sha256(file_path: Path) -> str:
    """Calculate SHA-256 cryptographic digest of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_matlab_file(file_path: Path) -> Dict[str, Any]:
    """Parse MATLAB file with scipy.io.loadmat and inspect sensor channels."""
    try:
        mat_dict = scipy.io.loadmat(file_path)
        valid_keys = [k for k in mat_dict.keys() if not k.startswith("__")]

        channels = {}
        for k in valid_keys:
            arr = mat_dict[k]
            channels[k] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        return {
            "is_valid_matlab": True,
            "keys": valid_keys,
            "channels": channels,
            "error": None,
        }
    except Exception as e:
        return {
            "is_valid_matlab": False,
            "keys": [],
            "channels": {},
            "error": str(e),
        }


def acquire_cwru_dataset(output_dir: str = "data/raw/cwru") -> Dict[str, Any]:
    """Download official CWRU MATLAB files, verify integrity, and generate manifest."""
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    manifest: List[Dict[str, Any]] = []

    total_files = sum(len(items) for items in DATASET_SPEC.values())
    print(f"=== InspectAI CWRU Dataset Acquisition Pipeline ===")
    print(f"Target Directory: {dest_dir.resolve()}")
    print(f"Total Files Scheduled: {total_files}\n")

    for category, items in DATASET_SPEC.items():
        print(f"--- Category: {category} ---")
        for item in items:
            filename = item["filename"]
            target_path = dest_dir / filename
            url = f"{OFFICIAL_BASE_URL}/{filename}"

            record = {
                "filename": filename,
                "category": category,
                "fault_type": item["fault_type"],
                "load_hp": item["load_hp"],
                "rpm": item["rpm"],
                "diameter_inch": item["diameter_in"],
                "source_url": url,
                "status": "PENDING",
                "file_size_bytes": 0,
                "sha256": None,
                "is_valid_matlab": False,
                "mat_keys": [],
                "channels": {},
                "error": None,
            }

            # Download if not already present or if empty
            if not target_path.exists() or target_path.stat().st_size == 0:
                print(f"Downloading {filename} from {url}...")
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        content = resp.read()
                        if len(content) < 5000:
                            raise ValueError(f"Downloaded payload suspiciously small ({len(content)} bytes)")
                        with open(target_path, "wb") as f_out:
                            f_out.write(content)
                    print(f"  -> Downloaded {len(content):,} bytes.")
                except Exception as e:
                    print(f"  -> ERROR downloading {filename}: {e}")
                    record["status"] = "DOWNLOAD_FAILED"
                    record["error"] = str(e)
                    manifest.append(record)
                    continue
            else:
                print(f"File {filename} already exists ({target_path.stat().st_size:,} bytes). Validating...")

            # Validate file properties
            file_size = target_path.stat().st_size
            record["file_size_bytes"] = file_size
            record["sha256"] = compute_sha256(target_path)

            mat_info = inspect_matlab_file(target_path)
            record["is_valid_matlab"] = mat_info["is_valid_matlab"]
            record["mat_keys"] = mat_info["keys"]
            record["channels"] = mat_info["channels"]

            if mat_info["is_valid_matlab"]:
                record["status"] = "SUCCESS"
                print(f"  -> SUCCESS: Valid MATLAB file. Keys: {mat_info['keys']}")
            else:
                record["status"] = "CORRUPT_MATLAB"
                record["error"] = mat_info["error"]
                print(f"  -> ERROR: Invalid MATLAB file structure: {mat_info['error']}")

            manifest.append(record)
            time.sleep(0.2)  # Respectful download spacing

    # Persist JSON manifest
    manifest_json_path = dest_dir / "download_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f_json:
        json.dump(manifest, f_json, indent=2)

    # Persist CSV manifest
    manifest_csv_path = dest_dir / "download_manifest.csv"
    with open(manifest_csv_path, "w", encoding="utf-8") as f_csv:
        f_csv.write("filename,category,fault_type,load_hp,rpm,diameter_inch,status,file_size_bytes,sha256,is_valid_matlab,source_url\n")
        for r in manifest:
            f_csv.write(
                f"{r['filename']},{r['category']},{r['fault_type']},{r['load_hp']},{r['rpm']},"
                f"{r['diameter_inch']},{r['status']},{r['file_size_bytes']},{r['sha256']},"
                f"{r['is_valid_matlab']},{r['source_url']}\n"
            )

    success_count = sum(1 for r in manifest if r["status"] == "SUCCESS")
    print(f"\n=======================================================")
    print(f"Acquisition Summary: {success_count}/{total_files} successfully verified.")
    print(f"JSON Manifest saved to: {manifest_json_path}")
    print(f"CSV Manifest saved to:  {manifest_csv_path}")
    print(f"=======================================================\n")

    return {
        "total_files": total_files,
        "success_count": success_count,
        "manifest_json": str(manifest_json_path),
        "manifest_csv": str(manifest_csv_path),
        "records": manifest,
    }


if __name__ == "__main__":
    acquire_cwru_dataset()

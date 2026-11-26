"""Script to acquire the 4-bearing Paderborn pilot dataset from Zenodo repository.

Downloads:
- K001.rar (Healthy / Normal)
- KA04.rar (Outer Race Fault - Accelerated Lifetime Fatigue)
- KI04.rar (Inner Race Fault - Accelerated Lifetime Fatigue)
- KB23.rar (Rolling Element / Ball Fault - Combined Fatigue)
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ZENODO_BASE = "https://zenodo.org/records/15845309/files"

PILOT_TARGETS = {
    "K001": {
        "class": "Normal",
        "archive_name": "K001.rar",
        "url": f"{ZENODO_BASE}/K001.rar?download=1",
        "expected_bytes": 173881721,
        "description": "Healthy reference bearing run-in > 50h",
    },
    "KA04": {
        "class": "Outer Race Fault",
        "archive_name": "KA04.rar",
        "url": f"{ZENODO_BASE}/KA04.rar?download=1",
        "expected_bytes": 180920703,
        "description": "Outer ring fatigue spall damage from lifetime test",
    },
    "KI04": {
        "class": "Inner Race Fault",
        "archive_name": "KI04.rar",
        "url": f"{ZENODO_BASE}/KI04.rar?download=1",
        "expected_bytes": 172644279,
        "description": "Inner ring fatigue spall damage from lifetime test",
    },
    "KB23": {
        "class": "Ball Fault",
        "archive_name": "KB23.rar",
        "url": f"{ZENODO_BASE}/KB23.rar?download=1",
        "expected_bytes": 170933916,
        "description": "Combined fatigue with flaking on rolling elements",
    },
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_archive(bearing_id: str, info: dict, raw_dir: Path) -> Path:
    dest_path = raw_dir / info["archive_name"]
    url = info["url"]
    expected_size = info["expected_bytes"]

    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        print(f"[{bearing_id}] Archive already complete: {dest_path.name} ({dest_path.stat().st_size:,} bytes)", flush=True)
        return dest_path

    max_retries = 10
    for attempt in range(1, max_retries + 1):
        current_size = dest_path.stat().st_size if dest_path.exists() else 0
        if current_size == expected_size:
            break
        print(f"[{bearing_id}] Attempt {attempt}/{max_retries}: Resuming {dest_path.name} (current: {current_size/(1024*1024):.1f} MB / total: {expected_size/(1024*1024):.1f} MB)...", flush=True)
        cmd = [
            "curl.exe",
            "-k",
            "-L",
            "-C", "-",
            "--retry", "5",
            "--retry-delay", "3",
            "-o", str(dest_path),
            url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        actual_size = dest_path.stat().st_size if dest_path.exists() else 0
        if actual_size == expected_size:
            print(f"[{bearing_id}] Download verified complete: {actual_size:,} bytes", flush=True)
            return dest_path
        print(f"[{bearing_id}] Download interrupted at {actual_size:,}/{expected_size:,} bytes. Retrying...", flush=True)
        time.sleep(2)

    actual_size = dest_path.stat().st_size if dest_path.exists() else 0
    if actual_size != expected_size:
        raise ValueError(f"Size mismatch for {dest_path.name}: expected {expected_size}, got {actual_size}")

    return dest_path


def extract_archive(archive_path: Path, extract_dir: Path) -> list:
    bearing_extract_dir = extract_dir / archive_path.stem
    bearing_extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_mats = list(bearing_extract_dir.rglob("*.mat"))
    if len(extracted_mats) >= 80:
        print(f"[{archive_path.stem}] Already extracted ({len(extracted_mats)} .mat files found in {bearing_extract_dir})", flush=True)
        return extracted_mats

    print(f"Extracting {archive_path.name} to {bearing_extract_dir}...", flush=True)
    cmd = ["tar.exe", "-xf", str(archive_path), "-C", str(bearing_extract_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"tar extraction failed for {archive_path.name}: {res.stderr}")

    extracted_files = list(bearing_extract_dir.rglob("*.mat"))
    print(f"Extracted {len(extracted_files)} .mat files for {archive_path.stem}", flush=True)
    return extracted_files


def main():
    raw_dir = Path("data/external/raw")
    extract_dir = Path("data/external/extracted")
    manifest_dir = Path("outputs/experiments/phase14")

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    acquisition_records = {}

    for b_id, info in PILOT_TARGETS.items():
        print("=" * 60, flush=True)
        print(f"Processing Bearing {b_id} ({info['class']})", flush=True)
        archive_path = download_archive(b_id, info, raw_dir)
        print(f"Computing SHA256 for {archive_path.name}...", flush=True)
        sha256_hash = compute_sha256(archive_path)
        print(f"SHA256: {sha256_hash}", flush=True)
        extracted_files = extract_archive(archive_path, extract_dir)

        acquisition_records[b_id] = {
            "bearing_id": b_id,
            "fault_class": info["class"],
            "description": info["description"],
            "archive_filename": info["archive_name"],
            "archive_path": str(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_hash,
            "source_url": info["url"],
            "extracted_directory": str(extract_dir / b_id),
            "mat_files_count": len(extracted_files),
            "sample_mat_files": [f.name for f in extracted_files[:5]],
            "acquisition_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    manifest_path = manifest_dir / "acquisition_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(acquisition_records, f, indent=2)
    print("=" * 60, flush=True)
    print(f"Successfully generated acquisition manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

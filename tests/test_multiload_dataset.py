"""Tests for Phase 10.6 CWRU Multi-Load Dataset Expansion.

Verifies:
1. Exactly 40 raw CWRU recordings exist in data/raw/cwru/.
2. All 40 files have valid MATLAB structure and non-empty Drive End (DE) acceleration signals.
3. The 4 historical holdout files (97, 169, 185, 197) are strictly preserved and marked holdout.
4. Manifest files exist and have consistent counts, schemas, and values.
5. Multi-load distribution is perfectly balanced across classes and loads.
"""

import json
from pathlib import Path
import pytest
import scipy.io
import pandas as pd

RAW_DIR = Path("data/raw/cwru")
OUTPUT_DIR = Path("outputs/experiments/phase10_6")

HOLDOUT_IDS = ["97", "169", "185", "197"]
NEW_IDS = [
    "106", "107", "108", "170", "171", "172", "210", "211", "212",
    "119", "120", "121", "186", "187", "188", "223", "224", "225",
    "131", "132", "133", "198", "199", "200", "235", "236", "237"
]


def test_raw_files_count():
    """Verify exactly 40 raw CWRU files exist."""
    mat_files = list(RAW_DIR.glob("*.mat"))
    assert len(mat_files) == 40, f"Expected 40 .mat files, found {len(mat_files)}"


def test_all_new_recordings_exist():
    """Verify all 27 new multi-load recordings exist."""
    for rec_id in NEW_IDS:
        filepath = RAW_DIR / f"{rec_id}.mat"
        assert filepath.exists(), f"Missing newly acquired file {filepath}"
        assert filepath.stat().st_size > 1_000_000, f"File {filepath} is suspiciously small"


def test_holdout_files_intact():
    """Verify all 4 historical holdout files exist with expected attributes."""
    for rec_id in HOLDOUT_IDS:
        filepath = RAW_DIR / f"{rec_id}.mat"
        assert filepath.exists(), f"Holdout file {filepath} missing!"
        assert filepath.stat().st_size > 2_000_000, f"Holdout file {filepath} corrupted!"


def test_all_recordings_have_valid_de_signal():
    """Verify every recording in data/raw/cwru has a valid DE channel with >= 100k samples."""
    for mat_file in RAW_DIR.glob("*.mat"):
        mat = scipy.io.loadmat(str(mat_file))
        data_keys = [k for k in mat.keys() if not k.startswith("__")]
        de_keys = [k for k in data_keys if "_DE_" in k or k.endswith("_DE")]
        assert len(de_keys) > 0, f"No DE channel found in {mat_file.name}: keys={data_keys}"
        de_data = mat[de_keys[0]].flatten()
        assert len(de_data) >= 120_000, f"DE signal too short in {mat_file.name}: {len(de_data)}"


def test_manifest_and_metadata_deliverables_exist():
    """Verify all Phase 10.6 output deliverables exist and are valid."""
    manifest_csv = OUTPUT_DIR / "cwru_40_recordings_manifest.csv"
    manifest_json = OUTPUT_DIR / "cwru_40_recordings_manifest.json"
    metadata_csv = OUTPUT_DIR / "cwru_40_metadata.csv"
    audit_json = OUTPUT_DIR / "dataset_integrity_audit.json"
    dist_json = OUTPUT_DIR / "distribution_summary.json"
    sha_json = OUTPUT_DIR / "sha256_inventory.json"
    sha_txt = OUTPUT_DIR / "sha256_inventory.txt"
    
    assert manifest_csv.exists()
    assert manifest_json.exists()
    assert metadata_csv.exists()
    assert audit_json.exists()
    assert dist_json.exists()
    assert sha_json.exists()
    assert sha_txt.exists()

    df_manifest = pd.read_csv(manifest_csv)
    assert len(df_manifest) == 40
    
    # Holdout flag check in manifest
    holdout_subset = df_manifest[df_manifest["recording_id"].astype(str).isin(HOLDOUT_IDS)]
    assert len(holdout_subset) == 4
    assert (holdout_subset["is_holdout"] == True).all()
    assert (holdout_subset["split"] == "test").all()

    # Non-holdout flag check
    non_holdout = df_manifest[~df_manifest["recording_id"].astype(str).isin(HOLDOUT_IDS)]
    assert len(non_holdout) == 36
    assert (non_holdout["is_holdout"] == False).all()


def test_distribution_balance():
    """Verify class and load balance in distribution summary."""
    dist_json = OUTPUT_DIR / "distribution_summary.json"
    with open(dist_json) as f:
        dist = json.load(f)
        
    assert dist["total_recordings"] == 40
    assert dist["historical_count"] == 13
    assert dist["newly_acquired_count"] == 27
    assert dist["holdout_count"] == 4
    
    # 10 recordings per load
    assert dist["load_distribution"]["0"] == 10
    assert dist["load_distribution"]["1"] == 10
    assert dist["load_distribution"]["2"] == 10
    assert dist["load_distribution"]["3"] == 10
    
    # Classes: 12 IR, 12 Ball, 12 OR, 4 Normal
    assert dist["class_distribution"]["Inner Race Fault"] == 12
    assert dist["class_distribution"]["Ball Fault"] == 12
    assert dist["class_distribution"]["Outer Race Fault"] == 12
    assert dist["class_distribution"]["Normal"] == 4


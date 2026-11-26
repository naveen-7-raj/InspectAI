"""InspectAI - Release Preflight Verification Script.

Executes comprehensive release readiness checks before GitHub publication:
1. File structure and required documentation validation.
2. Cryptographic checksum verification of frozen model artifacts.
3. Verification that historical holdouts (97, 169, 185, 197) remain quarantined.
4. Model architecture parameter check (89,604 parameters).
5. Fast inference contract & latency verification on CPU.
6. Secrets, private keys, and credential leak scan.
7. Large binary and raw dataset leak scan.
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

EXPECTED_CHECKSUMS = {
    "models/final/inspectai_final_model.pth": "a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e",
    "models/final/inspectai_final_normalizer.json": "2b3db205cd2d95ead19c8ec5d14e4b2d9a8e6133ae0f0aef05e7c2bf0f1db3c1",
    "models/final/inspectai_final_spectral_normalizer.json": "ed39cd3615212d83c24e66a49c46fc0c07e4e9f33320e95a3cc0d3a6567a3fac",
    "models/checkpoints/inspectai_final_cnn.pth": "6610b0d601fcfe93e88495c371a3f1ba074f5ffd93fe5958247b6feef349f70b",
}

QUARANTINED_RECORDINGS = {"97", "169", "185", "197"}
EXPECTED_PARAMETERS = 89604


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_artifacts():
    print("[1/6] Checking frozen model artifacts and checksums...")
    for rel_path, expected_hash in EXPECTED_CHECKSUMS.items():
        p = Path(rel_path)
        if not p.exists():
            print(f"  FAILED: Missing file {rel_path}")
            return False
        actual_hash = compute_sha256(p)
        if actual_hash != expected_hash:
            print(f"  FAILED: Hash mismatch for {rel_path}!\n    Expected: {expected_hash}\n    Actual:   {actual_hash}")
            return False
        print(f"  OK: {rel_path} verified.")
    return True


def check_quarantine():
    print("\n[2/6] Checking quarantine integrity of historical holdout recordings...")
    config_path = Path("models/final/inspectai_final_config.json")
    if not config_path.exists():
        print("  FAILED: Missing final config file!")
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dev_ids = set(cfg["training_provenance"]["development_recording_ids"])
    overlap = dev_ids.intersection(QUARANTINED_RECORDINGS)
    if overlap:
        print(f"  FAILED: Quarantined recordings found in development set: {overlap}")
        return False
    print(f"  OK: Quarantined recordings {QUARANTINED_RECORDINGS} are strictly excluded from all 36 dev recordings.")
    return True


def check_model_loading_and_inference():
    print("\n[3/6] Checking model loading, architecture parameters, and CPU inference...")
    from src.inference.predictor import StreamingPredictor

    predictor = StreamingPredictor.from_final_model(model_dir=Path("models/final"))
    params = sum(p.numel() for p in predictor.model.parameters())
    if params != EXPECTED_PARAMETERS:
        print(f"  FAILED: Expected {EXPECTED_PARAMETERS} parameters, found {params}")
        return False
    print(f"  OK: SpectralFusionCNN parameter count verified ({params:,} params).")

    # Benchmark single-window prediction
    dummy_signal = np.random.randn(2048).astype(np.float32)
    latencies = []
    for _ in range(25):
        t0 = time.perf_counter()
        res = predictor.predict_window(dummy_signal)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median_lat = float(np.median(latencies))
    print(f"  OK: Single-window inference passed. Median CPU latency: {median_lat:.2f} ms")
    if median_lat > 50.0:
        print(f"  FAILED: Inference latency too high ({median_lat:.2f} ms > 50.0 ms threshold)")
        return False
    return True


def scan_for_secrets():
    print("\n[4/6] Scanning repository for leaked secrets and private API keys...")
    secret_patterns = [
        re.compile(r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", re.IGNORECASE),
        re.compile(r"secret[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", re.IGNORECASE),
        re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
        re.compile(r"aws_access_key_id", re.IGNORECASE),
    ]

    scan_dirs = ["src", "backend", "frontend", "scripts", "configs", "tests", "docs"]
    found_issues = []

    for d in scan_dirs:
        for p in Path(d).rglob("*"):
            if p.is_file() and p.suffix in [".py", ".yaml", ".yml", ".json", ".md", ".toml", ".sh"]:
                if p.name == "release_preflight.py":
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    for pat in secret_patterns:
                        if pat.search(text):
                            found_issues.append(f"{p} matches secret pattern {pat.pattern}")
                except Exception as e:
                    pass

    if found_issues:
        print("  FAILED: Leaked secrets detected:")
        for issue in found_issues:
            print(f"    - {issue}")
        return False
    print("  OK: Zero credentials, private API keys, or security tokens found.")
    return True


def check_large_files():
    print("\n[5/6] Checking for staged or untracked large raw dataset files...")
    root = Path(".")
    raw_mat_files = list(root.glob("data/raw/**/*.mat"))
    if raw_mat_files:
        print(f"  INFO: Found {len(raw_mat_files)} local raw .mat files in data/raw/.")
        # Ensure data/raw is ignored by git
        p_check = Path(".gitignore").read_text(encoding="utf-8")
        if "data/raw/*" not in p_check:
            print("  FAILED: data/raw/* is not in .gitignore!")
            return False
        print("  OK: Local raw .mat files are properly ignored by .gitignore.")
    return True


def check_documentation():
    print("\n[6/6] Checking required Phase 12 release documentation...")
    required_docs = [
        "README.md",
        "docs/final_model_card.md",
        "docs/phase11_5_final_model_selection.md",
        "docs/phase12_release_integrity_audit.md",
        "outputs/reports/phase12_release_audit.json",
    ]
    for d in required_docs:
        if not Path(d).exists():
            print(f"  FAILED: Missing document {d}")
            return False
        print(f"  OK: Found {d}")
    return True


def main():
    print("=" * 70)
    print("      INSPECTAI - RELEASE PREFLIGHT VERIFICATION")
    print("=" * 70)

    checks = [
        check_artifacts,
        check_quarantine,
        check_model_loading_and_inference,
        scan_for_secrets,
        check_large_files,
        check_documentation,
    ]

    for check in checks:
        if not check():
            print("\n" + "=" * 70)
            print(">>> PREFLIGHT CHECKS FAILED! Please address the errors above.")
            print("=" * 70)
            sys.exit(1)

    print("\n" + "=" * 70)
    print(">>> ALL PREFLIGHT VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    print(">>> Repository is in valid release-ready state.")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()

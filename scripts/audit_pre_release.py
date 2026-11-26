"""Pre-release security, files, and integrity audit script."""
import os
import hashlib
import subprocess
from pathlib import Path

def main():
    print("=== 1. CHECKSUM VERIFICATION ===")
    hashes = {
        'models/final/inspectai_final_model.pth': 'a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e',
        'models/final/inspectai_final_normalizer.json': '2b3db205cd2d95ead19c8ec5d14e4b2d9a8e6133ae0f0aef05e7c2bf0f1db3c1',
        'models/final/inspectai_final_spectral_normalizer.json': 'ed39cd3615212d83c24e66a49c46fc0c07e4e9f33320e95a3cc0d3a6567a3fac'
    }

    all_matched = True
    for path_str, expected in hashes.items():
        p = Path(path_str)
        if not p.exists():
            print(f"MISSING: {p}")
            all_matched = False
            continue
        with open(p, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        match = actual == expected
        status = "MATCH" if match else "MISMATCH"
        print(f"  {p.name}: {status}")
        if not match:
            print(f"    Expected: {expected}")
            print(f"    Actual:   {actual}")
            all_matched = False

    print("\n=== 2. REQUIRED FILES AUDIT ===")
    required_paths = [
        "README.md",
        ".gitignore",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "src",
        "backend",
        "frontend",
        "configs",
        "scripts",
        "tests",
        "docs",
        "models/final/inspectai_final_model.pth",
        "models/final/inspectai_final_config.json",
        "models/final/inspectai_model_card.json",
        "docs/assets/cwru_baseline_prediction.png",
        "docs/assets/paderborn_adapted_prediction.png",
        "docs/assets/gradcam_temporal_attribution.png",
        "docs/assets/final_test_confusion_matrix.png",
        "docs/assets/final_training_curves.png",
        "docs/GITHUB_RELEASE_GUIDE.md",
        "docs/phase22_consistency_audit.md",
        "docs/phase22_release_checklist.md",
        "docs/phase22_release_report.md",
    ]
    missing_required = []
    for r in required_paths:
        exists = Path(r).exists()
        if not exists:
            missing_required.append(r)
        print(f"  {r}: {'EXISTS' if exists else 'MISSING'}")

    print("\n=== 3. SENSITIVE / LARGE FILES EXCLUSIONS ===")
    forbidden_extensions = ('.mat', '.rar', '.zip', '.tar', '.gz', '.npz', '.npy')
    unignored_forbidden = []
    for root, dirs, files in os.walk('.'):
        if '.git' in root or '.venv' in root or 'venv' in root:
            continue
        for f in files:
            if f.endswith(forbidden_extensions):
                rel = os.path.relpath(os.path.join(root, f), '.')
                # Check if ignored by git
                res = subprocess.run(["git", "check-ignore", "-q", rel])
                if res.returncode != 0:
                    unignored_forbidden.append(rel)
                    print(f"  WARNING: Unignored raw file: {rel}")

    if not unignored_forbidden:
        print("  All raw datasets (.mat, .rar, .npz, etc.) are 100% ignored by git.")

    print("\n=== 4. SUMMARY ===")
    print(f"Checksums: {'ALL PASS' if all_matched else 'FAIL'}")
    print(f"Required Files: {'ALL PRESENT' if not missing_required else f'MISSING: {missing_required}'}")
    print(f"Raw Files Ignored: {'CLEAN' if not unignored_forbidden else f'UNSAFE: {unignored_forbidden}'}")

if __name__ == "__main__":
    main()

"""InspectAI System & Installation Verification Utility.

Checks Python environment, dependencies, configuration, model checkpoints,
and authentic CWRU dataset files to ensure deployment readiness.
"""

import importlib
from pathlib import Path
import sys
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "yaml",
    "pydantic",
    "fastapi",
    "uvicorn",
    "websockets",
    "requests",
    "streamlit",
    "matplotlib",
    "pytest",
]

EXPECTED_CWRU_FILES = [
    "97.mat", "98.mat", "99.mat", "100.mat",
    "105.mat", "118.mat", "130.mat", "169.mat",
    "185.mat", "197.mat", "209.mat", "222.mat", "234.mat",
]


def check_environment() -> bool:
    print("=" * 70)
    print("InspectAI Environment & Artifact Verification")
    print("=" * 70)

    all_passed = True

    # 1. Python Version
    py_ver = sys.version_info
    print(f"\n[1] Python Environment: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 10):
        print("    [FAIL] Python >= 3.10 is required.")
        all_passed = False
    else:
        print("    [PASS] Python version compatible.")

    # 2. Package Availability
    print("\n[2] Checking Required Python Dependencies:")
    missing_packages = []
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"    - {pkg:<16}: {version}")
        except ImportError:
            print(f"    - {pkg:<16}: [MISSING]")
            missing_packages.append(pkg)
            all_passed = False

    if missing_packages:
        print(f"    [FAIL] Missing packages: {', '.join(missing_packages)}")
        print("    Fix: pip install -r requirements.txt")

    # 3. Model Checkpoints & Normalizer
    print("\n[3] Checking Frozen Model Artifacts:")
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    model_path = ckpt_dir / "inspectai_final_cnn.pth"
    norm_path = ckpt_dir / "inspectai_final_normalizer.json"

    if model_path.is_file():
        size_kb = model_path.stat().st_size / 1024.0
        print(f"    [PASS] CNN Checkpoint: {model_path.name} ({size_kb:.1f} KB)")
    else:
        print(f"    [FAIL] Missing CNN checkpoint at: {model_path}")
        all_passed = False

    if norm_path.is_file():
        print(f"    [PASS] Normalizer Spec: {norm_path.name}")
    else:
        print(f"    [FAIL] Missing normalizer specification at: {norm_path}")
        all_passed = False

    # 4. Authentic CWRU Dataset Files
    print("\n[4] Checking CWRU Raw Benchmark Recordings:")
    cwru_dir = PROJECT_ROOT / "data" / "raw" / "cwru"
    missing_files = []
    if not cwru_dir.is_dir():
        print(f"    [FAIL] CWRU directory missing at: {cwru_dir}")
        all_passed = False
    else:
        for fname in EXPECTED_CWRU_FILES:
            fpath = cwru_dir / fname
            if not fpath.is_file():
                missing_files.append(fname)

        if missing_files:
            print(f"    [FAIL] Missing {len(missing_files)} CWRU files: {', '.join(missing_files)}")
            all_passed = False
        else:
            print(f"    [PASS] All 13 authentic CWRU recordings present ({len(EXPECTED_CWRU_FILES)} files).")

    # 5. Model Forward Pass Smoke Test
    print("\n[5] Model Inference CPU Smoke Test:")
    try:
        from src.models.cnn_1d import VibrationCNN1D
        from src.utils.config_parser import load_config
        config = load_config()
        model = VibrationCNN1D.from_config(config)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_x = torch.randn(1, 1, 2048)
        with torch.no_grad():
            out = model(dummy_x)
        print(f"    [PASS] Forward pass successful. Output shape: {tuple(out.shape)}")
    except Exception as exc:
        print(f"    [FAIL] Model forward pass error: {exc}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("STATUS: ALL CHECKS PASSED. InspectAI is ready to run.")
        print("\nNext Steps:")
        print("  - Start FastAPI Backend:   uvicorn backend.main:app --port 8000")
        print("  - Start Dashboard:         streamlit run frontend/app.py --server.port 8501")
        print("  - Run Automated Tests:     pytest tests/ -v")
    else:
        print("STATUS: VERIFICATION FAILED. Review errors above.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)


# InspectAI GitHub Release & Publication Checklist

This document details the exact repository status, file tracking policies, and step-by-step procedures for the official GitHub release of **InspectAI**.

---

## 1. Repository State & Git Audit

- **Git Version**: `git version 2.55.0.windows.2` installed and operational.
- **Repository Initialization**: `.git` repository is currently uninitialized.
- **Branch / Remote State**: No local branches or remote remotes currently exist.
- **Secrets & Credentials Audit**: Confirmed zero API keys, secrets, tokens, or personal directory paths in source code or configurations.
- **Tracking Rule Enforcement**:
  - Raw binary MATLAB files (`data/raw/cwru/*.mat`, total ~58 MB) are strictly excluded via `.gitignore`.
  - Processed numerical window arrays (`data/processed/cwru_segmented_windows.npz`, ~23 MB) are strictly excluded.
  - PyTorch binary model checkpoints (`models/checkpoints/*.pth`, ~1.3 MB each) are strictly excluded.
  - Manifests (`download_manifest.csv`, `download_manifest.json`) and window split metadata (`window_split_metadata.csv`) are explicitly tracked to allow full reproducibility.

---

## 2. Inventory of Files for Commit

When `git add .` is executed with the current `.gitignore`, the following clean inventory will be staged:

### Tracked Source & Configuration Files
- `.github/workflows/tests.yml`: Multi-version CI workflow for Python 3.10, 3.11, 3.12.
- `.dockerignore`: Docker build context exclusion rules.
- `.env.example`: Non-sensitive environment variable configuration template.
- `.gitignore`: Hardened repository tracking rules.
- `docker-compose.yml`: Multi-service container orchestration blueprint.
- `Dockerfile`: CPU-optimized Python 3.11 container definition.
- `pytest.ini`: Test configuration with Windows exception safety.
- `README.md`: Master portfolio presentation documentation.
- `requirements.txt`: Python package dependency specifications.
- `backend/`: FastAPI application, modular routers (`health.py`, `prediction.py`, `streaming.py`), dependencies, and Pydantic schemas.
- `configs/config.yaml`: Centralized, reproducible parameter configuration.
- `data/raw/cwru/download_manifest.*`: Checksum-verified acquisition manifests.
- `data/processed/window_*metadata.csv`: Window partition and split metadata.
- `docs/`: Technical guides, architecture diagrams, video guides, and release notes.
- `docs/images/`: Curated technical diagrams and evaluation plots.
- `docs/video_assets/`: Assets for video editing (`model_evaluation_summary.png`, title cards).
- `frontend/app.py`: Interactive Streamlit industrial monitoring dashboard.
- `notebooks/`: Numbered validation and reproduction pipeline scripts (`02` through `10`, `verify_stream.py`).
- `scripts/`: Diagnostic verification tools (`verify_installation.py`, `verify_live_demo.py`, `generate_video_results_asset.py`).
- `src/`: Core Python modules for ingestion, preprocessing, modeling, training, inference, explainability, and streaming.
- `tests/`: Complete 15-module automated test suite (140 passing tests).

### Excluded Files (Protected by `.gitignore`)
- `data/raw/cwru/*.mat` (13 raw CWRU binary files)
- `data/processed/cwru_segmented_windows.npz` (heavy numerical array)
- `models/checkpoints/*.pth` (model weight binary checkpoints)
- `outputs/reports/*` (runtime CSV / JSON logs; reproduced on-demand)
- `outputs/figures/*` (reproduced on-demand by scripts)
- `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`

---

## 3. Step-by-Step GitHub Release Commands

To publish the repository to your GitHub account:

### Step 1: Initialize Git & Branch
```powershell
# Open Windows PowerShell in project root
cd path\to\inspectai

# Initialize local repository
git init -b main
```

### Step 2: Stage and Verify Staged Files
```powershell
# Stage all tracked files
git add .

# Verify status (ensure no .mat, .npz, or .pth files are staged)
git status
```

### Step 3: Create Initial Release Commit
```powershell
git commit -m "feat: InspectAI v0.1.0 - Explainable Industrial Health Monitoring System"
```

### Step 4: Link Remote & Push
```powershell
# Create an empty repository on GitHub named 'inspectai'
# Replace <YOUR_GITHUB_USERNAME> with your actual GitHub username:
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/inspectai.git

# Push to main branch
git push -u origin main
```

### Step 5: Create Release Tag
```powershell
# Tag the commit as v0.1.0
git tag -a v0.1.0 -m "InspectAI Release v0.1.0 - Initial Portfolio Release"
git push origin v0.1.0
```

---

## 4. Pre-Push Verification Checklist

- [x] All 140 automated tests pass (`python -m pytest tests/`).
- [x] Installation diagnostic passes (`python scripts/verify_installation.py`).
- [x] Live application smoke test passes (`python scripts/verify_live_demo.py`).
- [x] Zero personal paths (`C:\Users\...`) or secrets exist in tracked files.
- [x] All image paths in `README.md` resolve to valid files in `docs/images/`.
- [x] Holdout test evaluation results are honestly and accurately reported (Accuracy: 39.73%, Macro-F1: 34.04%).
- [x] Demo video placeholder in README is clearly marked as pending manual recording.

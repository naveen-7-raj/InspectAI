# InspectAI Artifact Distribution & Reproduction Guide

This guide details the acquisition, preparation, and distribution policies for the **Case Western Reserve University (CWRU)** vibration dataset and InspectAI neural network checkpoints.

---

## 1. Storage & Version Control Policy

To maintain clean repository hygiene and adhere to GitHub best practices:
- **Raw Sensor Data**: Large raw `.mat` files (~58 MB total across 13 files) are **excluded** from Git via `.gitignore`.
- **Processed Window Arrays**: The segmented numerical array (`data/processed/cwru_segmented_windows.npz`, ~23 MB) is **excluded** from Git.
- **Model Checkpoints**: PyTorch weight binaries (`models/checkpoints/*.pth`, ~1.3 MB each) are **excluded** from Git.
- **Manifests & Metadata**: Verified checksum manifests (`download_manifest.csv`, `download_manifest.json`) and recording-level split allocations (`window_split_metadata.csv`) are **tracked** in Git.

---

## 2. Dataset Acquisition Source & Procedure

### Official CWRU Source
- **Repository**: Case Western Reserve University Bearing Data Center
- **Website**: [https://engineering.case.edu/bearingdatacenter/download-data-file](https://engineering.case.edu/bearingdatacenter/download-data-file)
- **Experimental Rig**: 2 HP Reliance Electric induction motor, drive-end deep groove ball bearings (SKF 6205-2RS JEM), 12 kHz and 48 kHz sampling rates.

### Automated Acquisition
Execute the automated downloader from the project root:
```bash
python src/ingestion/download_cwru.py
```
This script downloads all 13 authentic benchmark recordings directly into `data/raw/cwru/`, validates internal MATLAB keys (`X..._DE_time`), and checks byte lengths against `data/raw/cwru/download_manifest.csv`.

---

## 3. End-to-End Pipeline Reproduction

If cloning the repository without pre-existing processed data or model checkpoints, run the following sequential commands to generate all artifacts from scratch:

```bash
# Step 1: Download authentic CWRU benchmark recordings
python src/ingestion/download_cwru.py

# Step 2: Validate preprocessing (anti-aliasing, DC removal, 4th-order Butterworth bandpass)
python notebooks/03_preprocessing_validation.py

# Step 3: Segment continuous waveforms into 2,048-sample windows (75% overlap)
python notebooks/04_windowing_validation.py
# -> Generates: data/processed/cwru_segmented_windows.npz
# -> Generates: data/processed/window_metadata.csv

# Step 4: Generate leakage-safe recording-level dataset splits (9 Dev, 4 Holdout Test)
python notebooks/05_dataset_split_validation.py
# -> Generates: data/processed/window_split_metadata.csv

# Step 5: Fit normalizer & train final 1D CNN model (4 fixed epochs)
python notebooks/09_final_train_and_test.py
# -> Generates: models/checkpoints/inspectai_final_cnn.pth
# -> Generates: models/checkpoints/inspectai_final_normalizer.json
# -> Evaluates on holdout test partition (818 windows) and exports reports

# Step 6: Generate 1D Grad-CAM attribution figures
python notebooks/10_generate_gradcam_explanations.py
```

---

## 4. Required Artifacts & Feature Dependency Matrix

The table below details which application capabilities depend on local artifacts:

| Application Feature | Required Artifacts | Behavior if Artifact is Missing |
| :--- | :--- | :--- |
| **Data Ingestion Unit Tests** | `configs/config.yaml` | Operates fully on synthetic in-memory fixtures. |
| **Model Architecture Tests** | `src/models/cnn_1d.py` | Operates fully without checkpoints (random initialization). |
| **FastAPI `/api/v1/health`** | None | Returns `{"status": "degraded", "model_loaded": false}` with a clear startup warning. |
| **FastAPI `/api/v1/predict`** | `inspectai_final_cnn.pth`, `inspectai_final_normalizer.json` | Returns `HTTP 503 Service Unavailable: Model artifacts not loaded`. |
| **WebSocket Replay** | Raw `.mat` file for requested recording | Closes WebSocket with `WS 1008 Policy Violation: Recording not found`. |
| **Streamlit Dashboard** | Dependent on FastAPI backend | Renders UI, but displays status banner: `Backend Degraded: Model not loaded`. |

---

## 5. Artifact Verification Diagnostic

To verify whether your local environment has the required dataset files and model checkpoints installed:
```bash
python scripts/verify_installation.py
```
This utility checks:
1. Python version (3.10+)
2. Required packages (`torch`, `fastapi`, `streamlit`, `scipy`, etc.)
3. `models/checkpoints/inspectai_final_cnn.pth`
4. `models/checkpoints/inspectai_final_normalizer.json`
5. Presence of all 13 CWRU `.mat` files in `data/raw/cwru/`
6. PyTorch forward pass smoke test on CPU


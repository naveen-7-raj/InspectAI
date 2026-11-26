# InspectAI — GitHub Release & Developer Reproduction Guide

**Version**: `v1.0.0`  
**Target Environment**: Windows, Linux, macOS (CPU-first, Python 3.10–3.14)  
**Hardware Requirements**: Standard x86_64 or ARM64 workstation (No GPU or CUDA required)

---

## 1. Quickstart Guide (Running Out of the Box)

InspectAI is distributed with the pre-trained, production-verified `SpectralFusionCNN` checkpoint (`models/final/inspectai_final_model.pth`, 368 KB) and frozen normalizers. **You do NOT need to download raw datasets or train models to run the system.**

### Step 1: Clone the Repository
```bash
git clone https://github.com/naveen-7-raj/InspectAI.git
cd InspectAI
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the System Verification Suite
Verify that all model contracts, streaming pipelines, and unit tests pass on your machine:
```bash
pytest tests/ -q
```
*Expected output: `300 passed in ~40s`.*

---

## 2. Launching the Services

InspectAI features a dual-service architecture: a FastAPI backend engine and an interactive Streamlit engineering dashboard.

### Option A: Local Execution (Recommended for Fast Exploration)

Open two terminal windows:

**Terminal 1 — FastAPI Backend Engine:**
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Prometheus metrics: [http://127.0.0.1:8000/metrics](http://127.0.0.1:8000/metrics)

**Terminal 2 — Streamlit Condition Monitoring Dashboard:**
```bash
streamlit run frontend/app.py
```
- Web Dashboard: [http://localhost:8501](http://localhost:8501)

### Option B: Docker Compose Deployment

> [!NOTE]
> Docker configuration files are fully prepared and hardened with non-root security (`appuser:1000`) and read-only volume mounts (`:ro`). Note that direct Docker execution was verified via static container audits during release staging.

```bash
# Build and start services in the background
docker-compose up --build -d

# View service logs
docker-compose logs -f

# Shutdown services
docker-compose down
```

---

## 3. Dataset Acquisition & Research Reproduction

The production checkpoint is frozen and bundled in `models/final/`. If you wish to reproduce historical cross-load training or cross-dataset domain adaptation experiments from scratch, follow these instructions to acquire the external datasets.

### 3.1 Case Western Reserve University (CWRU) Bearing Dataset
- **Official Source**: Case Western Reserve University Bearing Data Center
- **Access URL**: [https://engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter)
- **Licence**: Public academic research use with appropriate scholarly citation.
- **Acquisition Script**:
  ```bash
  python scripts/acquire_multiload_cwru.py
  ```
  This script downloads the 40 standard 12 kHz Drive End accelerometer recordings (36 development, 4 protected holdout) directly into `data/raw/cwru/`.

### 3.2 Paderborn University Bearing Dataset
- **Official Source**: Paderborn University Chair of Design and Drive Technology (KAt)
- **Access URL**: [https://mb.uni-paderborn.de/kat/forschung/datensaetze/lagerdaten/](https://mb.uni-paderborn.de/kat/forschung/datensaetze/lagerdaten/) / Zenodo deposit
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Bearings Used in Pilot**: `K001` (Healthy), `KA04` (Outer Race), `KI04` (Inner Race), `KB23` (Ball / rolling element).
- **Acquisition Script**:
  ```bash
  python scripts/acquire_paderborn_pilot.py
  ```
  This script verifies and extracts pilot files into `data/external/`.

---

## 4. Key CLI Scripts & Workflows

| Script | Purpose | Execution Command |
| :--- | :--- | :--- |
| **Release Preflight** | Verifies model hashes, normalizers, and configs | `python scripts/release_preflight.py` |
| **Technical Audit** | Inspects parameter counts, latency, and metrics | `python scripts/audit_technical_consistency.py` |
| **Cross-Load Generalization** | Evaluates CWRU leave-one-load-out splits | `python scripts/run_multiload_generalization.py` |
| **Paderborn Pilot Evaluation** | Evaluates zero-shot baseline on Paderborn | `python scripts/run_real_zero_shot.py` |
| **CORAL Disjoint Validation** | Evaluates recording-disjoint domain adaptation | `python scripts/run_phase17_experiments.py` |
| **End-to-End Validation** | Runs full test pipeline & API smoke tests | `python scripts/run_phase20_validation.py` |

---

## 5. Repository Structure

```
InspectAI/
├── backend/                  # FastAPI high-throughput REST & WebSocket service
│   ├── routes/               # API endpoints (health, predict, stream, metrics)
│   ├── dependencies.py       # Thread-safe model predictor singleton
│   ├── main.py               # FastAPI application entrypoint
│   └── schemas.py            # Pydantic v2 validation contracts
├── frontend/                 # Streamlit engineering dashboard
│   ├── app.py                # Multi-tab monitoring & explanation UI
│   └── components/           # Visualization widgets (Grad-CAM, FFT, audio)
├── models/
│   └── final/                # FROZEN PRODUCTION ASSETS (Immutable)
│       ├── inspectai_final_model.pth             # Model checkpoint (89,604 params, 368 KB)
│       ├── inspectai_final_normalizer.json       # Waveform normalizer
│       ├── inspectai_final_spectral_normalizer.json # Hilbert spectral normalizer
│       ├── inspectai_final_config.json           # Model configuration
│       └── inspectai_model_card.json             # Production model card
├── src/                      # Core algorithms and ML components
│   ├── explainability/       # 1D Temporal & Spectral Grad-CAM
│   ├── features/             # Hilbert envelope & FFT spectral extraction
│   ├── inference/            # StreamingPredictor, DomainAdapter (CORAL), DriftMonitor
│   ├── models/               # SpectralFusionCNN architecture
│   ├── monitoring/           # Prometheus metrics & statistical trackers
│   └── training/             # Normalizers and splitting protocols
├── tests/                    # 300 automated unit, integration, and streaming tests
├── docs/                     # Comprehensive engineering & experimental reports
├── configs/                  # Pipeline and model YAML configurations
├── Dockerfile                # Non-root, hardened CPU container
├── docker-compose.yml        # Multi-service container orchestration
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Project overview and architecture documentation
```

---

## 6. Open-Source Licensing Status

> [!IMPORTANT]
> **Licence Decision Pending**: An official open-source license (such as MIT or Apache-2.0) has not yet been selected by the project owner. Until the project owner attaches an official license, all code and architectural designs in this repository remain all rights reserved. Contributions and redistribution terms will be clarified upon publication of the chosen license.

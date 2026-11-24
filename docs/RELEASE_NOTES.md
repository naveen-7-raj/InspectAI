# InspectAI Release Notes – v0.1.0 (Initial Portfolio Release)

**Release Tag**: `v0.1.0`  
**Release Title**: `InspectAI v0.1.0 – Explainable Industrial Machine Health Monitoring System`  
**Date**: November 2026  
**Status**: Initial Portfolio Release for AI/ML Engineer Intern Technical Review  

---

## 1. Project Description for GitHub

> **Concise Repository Description (Under 350 Characters)**:  
> *Explainable industrial machine health monitoring platform for bearing fault diagnosis from high-frequency vibration signals. Features a lightweight 1D CNN (73k params), 1D Grad-CAM attribution, FastAPI REST/WebSocket telemetry, and Streamlit dashboard with leakage-safe evaluation on CWRU benchmarks.*

### Recommended GitHub Topics / Tags
`predictive-maintenance` • `vibration-analysis` • `1d-cnn` • `grad-cam` • `explainable-ai` • `machine-learning-engineering` • `fastapi` • `websockets` • `streamlit` • `pytorch` • `digital-signal-processing` • `cwru-dataset` • `industrial-ai` • `condition-monitoring`

---

## 2. Release Overview & Key Capabilities

**InspectAI** is an end-to-end machine learning engineering platform built to diagnose rotational bearing defects from high-frequency vibration accelerometer signals. Designed with production engineering standards and scientific rigor, InspectAI tackles common pitfalls in condition monitoring: eliminating overlapping-window data leakage and providing mechanical interpretability.

### Key Capabilities Included in v0.1.0:
1. **Digital Signal Conditioning**:
   - Polyphase anti-aliased resampling (normalizing 48 kHz normal baselines to 12 kHz).
   - Sensor DC bias subtraction ($\tilde{x}[n] = x[n] - \mu_x$).
   - 4th-order zero-phase forward-backward Butterworth bandpass filtering ($20\text{ Hz} - 5,000\text{ Hz}$).
2. **Leakage-Safe Recording-Level Splitting**:
   - Fixed 2,048-sample windows with 75% overlap (512-sample step).
   - Strict recording-level partitioning: all overlapping windows from a given recording reside exclusively in one partition.
   - 9 Development recordings (2,110 windows) vs. 4 Holdout test recordings (818 windows).
3. **Edge-Optimized 1D CNN**:
   - 3 Conv1D blocks with batch normalization, ReLU, max pooling, adaptive spatial pooling (length 4), and dropout (0.3).
   - Compact parameter footprint: **73,156 trainable parameters** (~292 KB).
   - Fast CPU inference latency: **~3 to 4 milliseconds per window**.
4. **One-Dimensional Grad-CAM Explainability**:
   - Feature activation gradients computed on the penultimate convolutional layer (`conv3`).
   - Linearly interpolated 1D attribution heatmaps ($[0.0, 1.0]$) isolating temporal impact intervals.
   - Diagnosed 21 periodic ground-truth impact spikes matching the Ball Pass Frequency of the Outer Race (BPFO ~105 Hz).
5. **FastAPI Telemetry Gateway**:
   - `GET /api/v1/health`: System health probe and model loading readiness.
   - `POST /api/v1/predict`: Signal preprocessing, 1D CNN classification, and optional Grad-CAM.
   - `WS /api/v1/stream/{recording_id}`: Simulated real-time sequential telemetry streaming replaying authentic CWRU recordings chunk-by-chunk.
6. **Streamlit Industrial Console**:
   - Real-time KPI summary cards, dynamic waveform visualizer, uncalibrated softmax distribution bars, and interactive Grad-CAM heatmap viewer.
7. **Production & Testing Hygiene**:
   - Complete 15-module automated test suite with **140 / 140 passing tests (100%)** running in ~12 seconds on CPU.
   - CPU-optimized Docker containerization (`Dockerfile`, `docker-compose.yml`).
   - GitHub Actions multi-version CI configuration (`.github/workflows/tests.yml`).

---

## 3. Scientific Disclosure & Final Holdout Results

InspectAI adheres to scientific honesty. When evaluated on the four untouched holdout recordings (`97` Normal, `185` Ball Fault 0.014", `169` Inner Race 0.014", `197` Outer Race 0.014"), the final frozen model achieved:

| Metric | Result | Evaluation Partition |
| :--- | :---: | :--- |
| **Overall Test Accuracy** | **39.73%** (325 / 818) | 4 Untouched Recordings |
| **Macro-Averaged F1** | **34.04%** | 4 Untouched Recordings |
| **Normal Baseline F1** | **90.27%** | Precision: 82.27%, Recall: 100.0% |
| **Ball Fault F1** | **45.88%** | Precision: 30.87%, Recall: 89.32% |
| **Inner Race Fault F1** | **0.00%** | Precision: 0.00%, Recall: 0.00% (Defaulted to Ball) |
| **Outer Race Fault F1** | **0.00%** | Precision: 0.00%, Recall: 0.00% (Defaulted to Ball) |

### Engineering Diagnosis
The model correctly classified 100% of Normal windows and 89.32% of Ball Fault windows, but defaulted 100% of unseen race windows to Ball Fault. This failure is hypothesized to result from **intermediate defect geometry domain shift** (training on 0.007" and 0.021" faults vs. testing on intermediate 0.014" faults) and **spatial load-zone orientation variance** (outer race defect centered at 6:00 in the primary load zone vs. 3:00 / 12:00 in training).

> **Deployment Caution**: The current model is an educational and portfolio demonstration. It is **not** certified or recommended for production industrial machinery safety decisions without domain adaptation and epistemic uncertainty gating.

---

## 4. Getting Started

### Local Installation
```bash
git clone https://github.com/[your-username]/inspectai.git
cd inspectai
python -m venv .venv
.venv\Scripts\Activate.ps1    # On Linux: source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/verify_installation.py
```

### Reproducing Data & Model Artifacts
```bash
python src/ingestion/download_cwru.py
python notebooks/04_windowing_validation.py
python notebooks/05_dataset_split_validation.py
python notebooks/09_final_train_and_test.py
```

### Running Test Suite
```bash
python -m pytest tests/ -v
```
*(All 140 automated tests pass on CPU).*


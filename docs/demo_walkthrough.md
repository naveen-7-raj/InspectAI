# InspectAI: Step-by-Step Operator & Recruiter Demo Walkthrough

This document outlines the step-by-step workflow for demonstrating InspectAI either live or via pre-recorded video. It follows the exact 3–5 minute demonstration narrative outlined in `docs/demo_script.md`.

---

## Prerequisites & Environment Setup

Ensure the Python environment is activated and dependencies are installed:
```powershell
# In PowerShell:
.\venv\Scripts\Activate.ps1

# Terminal 1: Launch FastAPI Gateway
$env:INSPECTAI_USE_FINAL_MODEL = "1"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Launch Streamlit Operator Console
streamlit run frontend/app.py --server.port 8501
```

Open the dashboard in your browser at `http://localhost:8501`.

---

## Step 1 — Dashboard Overview & Connection Verification

1. **Observe the Sidebar**:
   - Connection status indicator will display `🟢 Backend Online (SpectralFusionCNN)`.
   - Verify that the active model is `SpectralFusionCNN` running on `CPU`.
2. **Review KPI Metric Cards**:
   - Total Streamed Windows: `0`
   - Active Diagnosis: `STANDBY`
   - Pipeline Latency: `-- ms`

---

## Step 2 — Healthy Machine Telemetry (CWRU Recording 97)

1. **Select Recording**: In the sidebar dropdown, select `Normal_0_97.mat` (CWRU Drive-End healthy baseline motor at 1,797 RPM, 0 HP).
2. **Set Replay Speed**: Select `1.0x` (real-time $42.67\,\text{ms}$ interval) or `2.0x`.
3. **Inference Mode**: Leave set to `baseline`.
4. **Click "Start Telemetry Stream"**:
   - The live waveform canvas updates with smooth, low-amplitude Gaussian vibration ($< 0.1\,\text{g}$).
   - The diagnosis card displays `NORMAL` in green.
   - The softmax confidence bar indicates `> 95% Normal`.
   - Edge CPU latency displays $\approx 25 - 30\,\text{ms}$.

![Healthy Baseline Waveform](assets/cwru_baseline_prediction.png)

---

## Step 3 — Inner Race Fault Emergence (CWRU Recording 169)

1. **Click "Stop Telemetry Stream"**.
2. **Select Recording**: In the dropdown, select `169.mat` (0.007″ single-point defect on Inner Race at 1,797 RPM).
3. **Click "Start Telemetry Stream"**:
   - The waveform immediately displays characteristic, repetitive shock transients reaching $\pm 1.5\,\text{g}$.
   - The diagnosis transitions to `CRITICAL: INNER RACE FAULT` in red.
   - Softmax posterior probability rises to `> 98% Inner Race Fault`.

---

## Step 4 — Inspecting 1D Grad-CAM Physical Attribution

1. **Click "Explain Current Window (Grad-CAM)"**:
   - The Grad-CAM temporal inspector expands beneath the waveform.
   - Observe the saliency curve overlaid on the raw vibration time-series.
   - **Diagnostic Insight**: Saliency peaks align directly with the high-amplitude impact spikes. The interval between peaks is approximately $9.4\,\text{ms}$, which exactly matches the theoretical Ball Pass Frequency Inner Race ($\text{BPFI} = 156.1\,\text{Hz}$ at 1,797 RPM).
   - This proves the neural network is triggering on true impact mechanics rather than spurious background noise.

![1D Grad-CAM Temporal Heatmap](assets/gradcam_temporal_attribution.png)

---

## Step 5 — Cross-Rig Domain Collapse (Paderborn University K001)

1. **Switch to External Benchmark**: Select Paderborn recording `N09_M07_F10_K001_1.mat` (healthy bearing on an independent test rig with different sensor placement and radial load).
2. **Run in `baseline` Mode**:
   - Because the model was trained exclusively on CWRU, domain collapse occurs: the model misclassifies healthy Paderborn vibration as an inner or outer race fault.
   - Highlight this to the viewer as an honest demonstration of why academic cross-dataset zero-shot claims often fail in real industrial settings.

---

## Step 6 — Unsupervised Domain Adaptation (CORAL)

1. **Toggle Inference Mode**: Change Mode from `baseline` to `adaptive`.
2. **Check "Buffer Unlabeled Calibration Windows"** (`calibrate_first=True`).
3. **Start Telemetry Stream**:
   - The system buffers 50 unlabelled windows of the new rig's baseline running data.
   - In background CPU execution, `DomainAdapter` computes source covariance $C_S$ and target covariance $C_T$, calculating the whitening/coloring transformation $A_{\text{coral}} = C_T^{-1/2} C_S^{1/2}$.
   - The adapter status updates to `Calibrated (kappa = 35,481.5)`.
   - Adapted predictions immediately switch to **`Normal`** with `is_adapted=True`.

![Paderborn Adapted Prediction](assets/paderborn_adapted_prediction.png)

---

## Step 7 — Testing Fault-Suspect Recalibration Guardrails

1. **Simulate Defect Injection**:
   - In adaptive mode, simulate the emergence of a sudden bearing defect (high kurtosis $> 4.5$ or fault posterior $\ge 0.35$).
   - Observe the `DriftMonitor` status card transition to `FAULT_SUSPECT`.
   - **Safety Assertion**: An alert banner displays: `[SAFETY ALERT] Adaptation Suppressed due to Fault Suspicion`.
   - Confirm that the adapter does **not** update its covariance matrices. Mechanical damage is alerted to the maintenance crew, never adapted away.

---

## Step 8 — Scraping Prometheus Monitoring Telemetry

1. **Open New Browser Tab**: Navigate to `http://localhost:8000/metrics`.
2. **Verify Telemetry Exposition**:
   - Find `inspectai_predictions_total` with breakdown by `inference_mode` (`baseline` vs `adaptive`) and `predicted_class`.
   - Inspect `inspectai_inference_latency_seconds_bucket` histogram.
   - Verify `inspectai_adapter_condition_number` and `inspectai_domain_drift_frobenius` gauges.
3. **Conclude**: Point out that the system operates with $10.4\times$ real-time headroom and sub-30ms latency on standard CPU hardware.

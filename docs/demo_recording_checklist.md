# InspectAI: Video Demonstration Recording Checklist

This checklist guides the recording of a professional 2.5-to-3-minute video demonstration of **InspectAI** for recruiter and engineering portfolio evaluation.

---

## 1. Pre-Recording Environment Setup

- [ ] **Display Resolution**: Set screen resolution to **1920x1080 (1080p, 16:9)** with 100% OS display scaling.
- [ ] **Audio Setup**:
  - [ ] External microphone configured with noise suppression and pop filter.
  - [ ] Test recording completed: voice clear, no background fan or mechanical noise.
  - [ ] Room acoustics quiet.
- [ ] **Software & Tools**:
  - [ ] OBS Studio installed and configured:
    - Video: 1080p @ 30 FPS or 60 FPS (Base Canvas & Output Scaled: 1920x1080).
    - Bitrate: 6,000 kbps (CBR), NVENC or x264 encoder.
    - Audio: 48 kHz, 320 kbps AAC.
  - [ ] Web Browser (Chrome/Edge/Brave) zoomed to 100% or 110% for optimal readability.
  - [ ] Terminal window (PowerShell / Windows Terminal) styled with clean dark theme (Monokai or OneDark), font size 14pt+.
- [ ] **Repository State**:
  - [ ] Virtual environment activated (`.\venv\Scripts\activate`).
  - [ ] Preflight validation executed and verified:
    ```bash
    python scripts/release_preflight.py
    ```
  - [ ] Pytest suite verified:
    ```bash
    python -m pytest tests/ -q
    ```
    *Result must show 224 passed.*

---

## 2. Service Initialization

- [ ] **Terminal 1: Launch FastAPI Telemetry Backend**:
  ```bash
  uvicorn backend.main:app --host 127.0.0.1 --port 8000
  ```
  - Verify terminal output shows: `InspectAI model artifacts successfully initialized in lifespan.`
  - Open `http://127.0.0.1:8000/docs` in browser tab 1.
- [ ] **Terminal 2: Launch Streamlit Operator Console**:
  ```bash
  streamlit run frontend/app.py
  ```
  - Verify dashboard loads at `http://localhost:8501` in browser tab 2.
  - Check sidebar health indicator shows: **`Backend: Healthy (Model: Loaded)`**.

---

## 3. Scene-by-Scene Recording Execution Flow

Follow `docs/demo_script.md` during recording:

| Time | Scene | Screen Layout & Actions | Spoken Focus |
|---|---|---|---|
| **0:00 - 0:20** | **1. Introduction** | Browser: GitHub README header & badges. | Industrial bearing failure problem, data leakage challenge, black-box opacity. |
| **0:20 - 0:42** | **2. Preprocessing & Multi-Load** | Browser: Signal conditioning diagram & windowing section. | 40 authentic CWRU recordings, polyphase decimation, Butterworth filter, group-aware partitioning. |
| **0:42 - 1:08** | **3. SpectralFusionCNN Architecture** | Browser / Graphic: Dual-branch architecture diagram. | Hilbert envelope spectrum, demodulation, BPFI/BPFO invariant harmonics, 89,604 params, 2.5 ms CPU latency. |
| **1:08 - 1:32** | **4. LOLO Benchmark Results** | Browser: Multi-load LOLO validation matrix. | 99.81% Macro-F1 across unseen motor loads, 100% accuracy on recording 172. |
| **1:32 - 1:58** | **5. Live Dashboard Streaming** | Browser tab 2: Streamlit console. Select recording `118.mat` (Ball Fault) or `209.mat` (Inner Race). Click "Start Telemetry Stream". | Authentic 42.67 ms step replay, live waveform visualizer, KPI metric updates, latency counter. |
| **1:58 - 2:25** | **6. 1D Grad-CAM Explainability** | Browser tab 2: Enable "Generate 1D Grad-CAM Attribution". | Saliency spikes matching theoretical defect repetition periods (BPFO/BPFI), physical verification vs acoustic noise. |
| **2:25 - 2:45** | **7. Testing & Production Engineering** | Terminal 3: Run `pytest tests/ -q` (shows 224 passed). Switch to `/docs` OpenAPI tab. | 224 automated tests passing, clean FastAPI schemas, honest scope limitations (steady-state replay, no hardware DAQ). |
| **2:45 - 2:55** | **8. Conclusion & Outro** | Browser: Repository top, GitHub link, MIT License badge. | Summary of end-to-end ML engineering, invitation for questions and portfolio review. |

---

## 4. Post-Recording Quality Control

- [ ] **Pacing & Duration**: Total runtime between **2:30 and 3:00**.
- [ ] **Audio Clarity**: Voiceover is consistently audible without clipping or distortion; no long awkward pauses (> 3 seconds).
- [ ] **Video Quality**: Text, code fonts, and dashboard charts are crisp and readable at 1080p.
- [ ] **Honest Claims**: No claims of real physical sensor monitoring, early failure prediction, or RUL prognostics were made.
- [ ] **File Export**: Rendered to MP4 (H.264 / AAC) at 1080p, file size under 150 MB.
- [ ] **Upload**: Upload to YouTube (Unlisted or Public) or Google Drive, and embed URL in `README.md`.


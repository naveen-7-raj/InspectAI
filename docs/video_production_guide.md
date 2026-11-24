# InspectAI Video Production Guide

This guide provides instructions for capturing genuine, high-definition screen recordings of the **InspectAI** platform on Windows using OBS Studio.

---

## 1. Technical Recording Specifications

- **Display Resolution**: `1920 × 1080` (1080p Full HD, 16:9 aspect ratio).
- **Display Scaling**: Set Windows OS scaling to `100%` (Settings → Display → Scale: 100%) to prevent blur or pixel distortion.
- **Recording Frame Rate**: `30 FPS` (or `60 FPS` if system hardware allows).
- **Video Encoder**: `NVIDIA NVENC H.264` or `Intel QuickSync H.264` (Fallback: `x264`).
- **Bitrate**: `6,000 kbps` CBR with audio at `192 kbps AAC`.
- **Recommended Recording Software**: **OBS Studio** (v30+ free, open-source).

---

## 2. OBS Studio Setup on Windows

1. **Install OBS Studio**: Download from [obsproject.com](https://obsproject.com/).
2. **Create Scene**:
   - Add a **Display Capture** or **Window Capture** source targeting your primary web browser (Chrome, Edge, or Firefox).
   - Set browser zoom level to `100%`.
   - Close all non-relevant browser tabs, bookmarks bars, and taskbars for a clean visual appearance.
3. **Configure Audio**:
   - Add **Audio Input Capture** (your USB microphone or headset).
   - Apply a mild Noise Suppression filter and Compressor in OBS filter settings.

---

## 3. Application Startup Sequence

Open two separate Windows PowerShell or Command Prompt terminals:

### Terminal 1: Launch FastAPI Backend
```powershell
# Navigate to repository root and activate virtual environment
cd path\to\inspectai
.venv\Scripts\Activate.ps1

# Start Uvicorn ASGI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
- **Verify Startup**: Terminal should output:
  ```text
  [INFO] InspectAI.Backend: InspectAI backend artifacts successfully loaded and ready for inference.
  [INFO] Uvicorn running on http://0.0.0.0:8000
  ```

### Terminal 2: Launch Streamlit Frontend
```powershell
# Navigate to repository root and activate virtual environment
cd path\to\inspectai
.venv\Scripts\Activate.ps1

# Start Streamlit application
streamlit run frontend/app.py --server.port 8501
```
- **Verify Startup**: Browser should automatically open to `http://localhost:8501`.

---

## 4. Scene-by-Scene Screen Capture Instructions

### Scene A: Swagger OpenAPI Interface
1. **Target URL**: `http://localhost:8000/docs`
2. **Action in Browser**:
   - Show the header: `InspectAI - Explainable Industrial Health Monitoring API (v0.1.0)`.
   - Expand `GET /api/v1/health` and click **Try it out** → **Execute**. Show the `200 OK` response with `status: healthy` and `device: cpu`.
   - Expand `POST /api/v1/predict` to demonstrate the Pydantic schema with `signal`, `sampling_rate_hz`, and `explain`.
3. **Target Duration**: ~15–20 seconds.

---

### Scene B: Streamlit Dashboard Configuration
1. **Target URL**: `http://localhost:8501`
2. **Action in Browser**:
   - Highlight the sidebar:
     - **Select CWRU Recording**: Choose `118.mat (Ball Fault, 0.007", 0 HP)`.
     - **Replay Speed Multiplier**: Select `1.0× (Real-time)` or `2.0× (Fast)` (Use documented supported speeds).
     - **1D Grad-CAM Toggle**: Leave unchecked initially to demonstrate standard inference speed.
3. **Target Duration**: ~15 seconds.

---

### Scene C: Live Simulated Vibration Streaming
1. **Action in Browser**:
   - Click the green **Start Replay** button in the sidebar.
   - Observe the live dynamic waveform visualizer updating continuously.
   - Observe the KPI summary cards:
     - *Status*: `Active Simulation`
     - *Windows Processed*: Counting up (0 → 236)
     - *Inference Latency*: Showing sub-4 ms execution time
   - Point cursor to the **Uncalibrated Softmax Distribution** bars indicating near 100% confidence for `Ball Fault`.
2. **Target Duration**: ~30 seconds.

---

### Scene D: Real-Time 1D Grad-CAM Explainability
1. **Action in Browser**:
   - Pause the stream or start a new simulation with **Enable 1D Grad-CAM (Attribution Heatmap)** checked.
   - Scroll down to the **1D Grad-CAM Attribution Overlay** section.
   - Point to the red/yellow high-saliency impact regions overlaid onto the 2,048-sample waveform.
   - Explain how the activation heatmap isolates physical impact intervals.
2. **Target Duration**: ~25 seconds.

---

### Scene E: Terminal Test Suite Run
1. **Action in PowerShell**:
   - Open a clean terminal window and execute:
     ```powershell
     python -m pytest tests/
     ```
   - Record the clean output showing all `140 passed in ~12s`.
2. **Target Duration**: ~10–15 seconds.

---

## 5. Post-Recording Verification Checklist

Before editing the final video, verify each captured video clip:
- [ ] Screen resolution is 1920x1080 without letterboxing or black borders.
- [ ] Text in Streamlit and Swagger is crisp and easily readable.
- [ ] No personal bookmarks, taskbars, or notifications are visible.
- [ ] Replay speed selected is one of the supported values: `0.5x`, `1.0x`, `2.0x`, or `5.0x` (never internal debug values).
- [ ] Voice audio is clear with minimal background hiss or echo.
- [ ] Streamlit interface shows authentic CWRU signal telemetry.


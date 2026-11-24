# InspectAI: Streamlit Monitoring Dashboard User & Operator Guide (Phase 8.1)

## 1. Overview & System Capabilities

The **InspectAI Streamlit Monitoring Dashboard** provides an interactive, operator-facing interface for simulated real-time industrial bearing fault diagnosis and Explainable AI (1D Grad-CAM).

### Key Dashboard Features:
- **Real-Time Telemetry Streaming**: Replays authentic Case Western Reserve University (CWRU) vibration recordings via WebSockets at configurable speeds ($0.5\times$, $1\times$, $2\times$, $5\times$).
- **Live Waveform Display**: Plots the incoming 2,048-sample vibration window in physical acceleration units ($g$) across a 170.67 ms temporal span.
- **Uncalibrated Model Scores**: Displays live classification scores across all four bearing condition categories (Normal, Ball Fault, Inner Race Fault, Outer Race Fault).
- **1D Grad-CAM Attribution Panel**: On-demand activation of feature attribution heatmaps isolating exact temporal shock bursts influencing model decisions.
- **Historical Telemetry Log**: Maintains a sliding table of recent predictions with latency and timestamps.

---

## 2. Installation & Requirements

Ensure that the Python environment contains the required dependencies:
```bash
pip install fastapi uvicorn streamlit websockets requests matplotlib pandas numpy scipy torch
```

---

## 3. How to Launch the Application

The InspectAI platform operates as a decoupled architecture:
1. **FastAPI Backend**: Hosts the frozen 1D CNN model, normalizer, and WebSocket streaming engine.
2. **Streamlit Frontend**: Consumes WebSocket prediction packets and renders interactive visualizations.

### Step 1: Start the FastAPI Backend
In a terminal, run:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Verify the backend is operational by browsing to [http://localhost:8000/docs](http://localhost:8000/docs) or checking `GET http://localhost:8000/api/v1/health`.*

### Step 2: Start the Streamlit Dashboard
In a separate terminal, launch:
```bash
streamlit run frontend/app.py --server.port 8501
```
*The dashboard will automatically open in your default browser at [http://localhost:8501](http://localhost:8501).*

---

## 4. Operator Workflow: How to Monitor Replays

### Step 1: Verify Backend Connectivity
Look at the top sidebar badge:
- **`● Backend Connected: VibrationCNN1D`**: The backend is reachable and the model checkpoint is loaded on CPU.
- **`✕ Backend Offline`**: Verify that the FastAPI server is running on the specified URL.

### Step 2: Select a CWRU Recording
Choose one of the 13 authentic CWRU recordings from the sidebar dropdown:
- **Normal Baselines (48 kHz $\to$ 12 kHz)**: `97` (0 HP), `98` (1 HP), `99` (2 HP), `100` (3 HP).
- **Ball Faults (12 kHz)**: `118` (0.007"), `185` (0.014"), `222` (0.021").
- **Inner Race Faults (12 kHz)**: `105` (0.007"), `169` (0.014"), `209` (0.021").
- **Outer Race Faults (12 kHz)**: `130` (0.007"), `197` (0.014"), `234` (0.021").

### Step 3: Configure Replay Speed
- **$0.5\times$ (85.3 ms delay)**: Slower pace for careful waveform inspection.
- **$1.0\times$ (42.7 ms delay)**: Real-time physical sensor pacing (512 samples at 12 kHz).
- **$2.0\times$ (21.3 ms delay)**: Double-speed accelerated replay.
- **$5.0\times$ (8.5 ms delay)**: Rapid test replay.

### Step 4: Toggle Grad-CAM (Optional)
Check **"Enable 1D Grad-CAM"** if you wish to visualize convolutional feature gradient attributions for each window. *Note: Grad-CAM increases CPU inference latency from ~2 ms to ~12 ms per window.*

### Step 5: Start & Stop Controls
- Click **`▶ Start Replay`**: Begins chronological WebSocket streaming. The KPI cards, waveform chart, score bars, and history table will update dynamically.
- Click **`⏹ Stop Replay`**: Immediately disconnects the WebSocket stream. The dashboard retains the latest state and history without crashing.

---

## 5. Interpreting Prediction Scores & Grad-CAM

### Diagnostic Softmax Scores:
- The bar chart displays model softmax outputs for each class.
- **Important**: These scores are **uncalibrated model activations**. They reflect the network's comparative feature affinity, **not** certified statistical failure probabilities.
- As discovered in Phase 4 and 5 diagnostic evaluations, the network may exhibit high softmax confidence (>90%) on unseen intermediate defect sizes (0.014") even when misclassifying them as Ball Fault.

### 1D Grad-CAM Heatmap Interpretation:
- When enabled, the Grad-CAM curve ($0.0$ to $1.0$) shows which temporal points had the highest positive gradient impact on the target class logit in convolutional layer `conv3`.
- Peaks above the $0.6$ threshold indicate shock impulses or periodic bursts that triggered the CNN filters.
- **Physical Caution**: Grad-CAM shows **model mathematical attribution**, not direct physical proof of bearing mechanics.

---

## 6. Simulation Limitations & Boundaries

1. **Simulation Only**: All data streams originate from historical CWRU laboratory recordings. No live physical machinery is connected.
2. **No Remaining Useful Life (RUL)**: The system performs discrete fault classification per window; it does not estimate degradation speed or time-to-failure.
3. **No Safety-Critical Reliance**: Model predictions should be verified with conventional spectral vibration analysis (e.g. FFT demodulation, BPFO/BPFI tracking) before taking maintenance action.


# System Architecture & Technical Specification

## 1. Overview
The **InspectAI** platform is engineered to ingest raw accelerometer data from rotating machinery, apply digital signal processing techniques, perform fault detection using Convolutional Neural Networks (CNNs), and output both classification probabilities and spatial/temporal attributions explaining *why* a particular failure was detected.

---

## 2. Pipeline Subsystems

### 2.1 Ingestion (`src/ingestion/`)
- Interfaces with real vibration benchmark formats (e.g., MATLAB `.mat` files from Case Western Reserve University, CSV logs from NASA IMS bearing tests).
- Validates data integrity, channel alignment, and verifies timestamp monotonicity without modifying raw readings.

### 2.2 Preprocessing (`src/preprocessing/`)
- **Anti-aliasing and Bandpass Filtering**: Employs Butterworth filters (e.g., 50 Hz – 5,000 Hz) to eliminate baseline drift and high-frequency acoustic noise.
- **Sliding Window Segmentation**: Slices continuous vibration time-series into overlapping windows (e.g., $N = 2048$ samples with 75% overlap).
- **Signal Normalization**: Robust Z-Score standardization computed strictly over training sets to prevent data leakage.

### 2.3 Feature Extraction (`src/features/`)
- **Time-Domain Indicators**: Root Mean Square (RMS), Peak-to-Peak, Kurtosis, Skewness, Crest Factor, Shape Factor, Impulse Factor.
- **Frequency-Domain Transformations**: Fast Fourier Transform (FFT) spectrum extraction.
- **Time-Frequency Representations**: Short-Time Fourier Transform (STFT) and Continuous Wavelet Transform (CWT) generating 2D spectral energy matrices.

### 2.4 Model Architectures (`src/models/`)
- **1D Dilated CNN**: Directly ingests raw vibration windows ($1 \times 2048$). Uses multi-scale 1D convolutions with dilated receptive fields to capture transient shock impulses.
- **2D Spectrogram CNN**: Ingests 2D STFT/Scalogram representations to detect harmonic sidebands and frequency modulations associated with bearing defect frequencies.

### 2.5 Explainability Engine (`src/explainability/`)
- **Grad-CAM (1D / 2D)**: Gradients flowing back to the final convolutional feature maps are aggregated to identify critical temporal intervals or spectral resonance bands driving the prediction.
- **Integrated Gradients (Captum)**: Path-integral attributions highlighting sample-by-sample contribution compared against a neutral baseline (zero-vibration / healthy signal).

### 2.6 Streaming Simulation (`src/streaming/`)
- Realistic playback engine that streams real recorded test files in contiguous chunks over WebSockets at configurable sampling intervals, simulating industrial edge sensors.

### 2.7 Backend & Frontend (`backend/` & `frontend/`)
- **FastAPI Backend**: Exposes REST endpoints for model inference and a WebSocket server for low-latency live streaming.
- **Streamlit Frontend**: Displays live telemetry, health index gauges, FFT spectrums, and interactive Grad-CAM explainability heatmaps.


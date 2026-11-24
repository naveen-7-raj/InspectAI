# InspectAI Video Editing Text & Overlay Asset Guide

This document defines the on-screen text, title cards, lower-thirds, and visual styling for the InspectAI demonstration video.

---

## 1. Visual Styling Guide (Dark Industrial Theme)

- **Background Base**: `#0F172A` (Slate 900) or `#090D16` (Deep Charcoal)
- **Panel / Card Surface**: `#1E293B` (Slate 800) with subtle 1px border `#334155`
- **Primary Accent / Headers**: `#38BDF8` (Cyan / Sky 400)
- **Secondary Accent**: `#F59E0B` (Industrial Amber / Safety Orange)
- **Failure / Warning Highlight**: `#EF4444` (Crimson / Red 500)
- **Primary Text**: `#F8FAFC` (Slate 50, crisp white)
- **Secondary Text / Subtitles**: `#94A3B8` (Slate 400, muted grey)
- **Font Typography**:
  - Headers / Titles: Inter, Roboto, or Montserrat (Bold / Heavy)
  - Monospace / Metrics: JetBrains Mono or Fira Code

---

## 2. Title Cards & Lower-Third Overlays

### Title Card 1: Video Intro (0:00 - 0:05)
- **Main Header**: `INSPECTAI`
- **Sub-Header**: `Explainable Industrial Machine Health Monitoring System`
- **Presenter Name**: `[Your Name] | Aspiring AI/ML Engineer`
- **Technology Badges**: `PyTorch (CPU) • FastAPI • 1D Grad-CAM • Streamlit`

---

### Lower-Third 1: Problem Statement (0:20 - 0:30)
- **Top Line**: `THE INDUSTRIAL CHALLENGE`
- **Bottom Line**: `Bearing Failures Cause >40% of Rotating Machinery Breakdowns`

---

### Lower-Third 2: Signal Conditioning (0:38 - 0:48)
- **Top Line**: `SIGNAL CONDITIONING PIPELINE`
- **Bottom Line**: `Polyphase Anti-aliased Decimation to 12 kHz • 4th-Order Bandpass (20-5000 Hz)`

---

### Lower-Third 3: 1D CNN Architecture (0:58 - 1:08)
- **Top Line**: `EDGE-OPTIMIZED 1D CNN`
- **Bottom Line**: `73,156 Parameters • Receptive Fields 15 -> 7 -> 3 • Sub-4 ms CPU Inference`

---

### Lower-Third 4: Live Telemetry Console (1:18 - 1:28)
- **Top Line**: `SIMULATED LIVE MONITORING`
- **Bottom Line**: `Sequential WebSocket Telemetry • Authentic CWRU Replay (1.0x & 2.0x Speeds)`

---

### Lower-Third 5: 1D Grad-CAM Interpretability (2:00 - 2:12)
- **Top Line**: `EXPLAINABLE AI ATTRIBUTION`
- **Bottom Line**: `Penultimate Conv3 Gradients Isolate Ball Pass Impact Frequency (BPFO ~105 Hz)`

---

### Lower-Third 6: Recording-Level Splitting (2:22 - 2:32)
- **Top Line**: `LEAKAGE-SAFE EVALUATION`
- **Bottom Line**: `Strict Recording-Level Isolation • 140 Passing Automated Tests`

---

### Lower-Third 7: Scientific Disclosure & Limitations (2:40 - 2:50)
- **Top Line**: `HOLDOUT TEST EVALUATION`
- **Bottom Line**: `39.73% Accuracy • 34.04% Macro-F1 • Domain Shift Under Intermediate Defect Geometries`

---

### Closing Title Card: Outro (2:55 - 3:00)
- **Header**: `INSPECTAI`
- **Tagline**: `End-to-End Machine Learning Engineering for Industrial Diagnostics`
- **GitHub Repository**: `github.com/[your-username]/inspectai`
- **Contact / Portfolio**: `[Your Portfolio Website / LinkedIn URL]`
- **Data Citation**: `Case Western Reserve University (CWRU) Bearing Data Center`

---

## 3. Video Asset Manifest

The following authentic figures are curated in `docs/video_assets/` for video overlay:

1. `vibration_waveform_comparison.png`: 4-class conditioned vibration waveform comparisons.
2. `cnn1d_architecture.png`: Detailed layer-by-layer architectural diagram.
3. `recording_level_split.png`: Leakage-safe recording-level partition diagram.
4. `gradcam_misclassifications.png`: Diagnostic attribution overlay comparing predicted vs ground-truth impacts.
5. `final_test_confusion_matrix.png`: Holdout confusion matrix graphic.
6. `model_evaluation_summary.png`: Custom dark industrial graphic detailing split-by-split collapse and root-cause hypotheses.


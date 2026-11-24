# InspectAI: Video Editing Timeline & Production Blueprint

**Target Duration**: 2 minutes 50 seconds (170 seconds)  
**Frame Rate**: 30 FPS (or 60 FPS) at 1920x1080  
**Audio Mix**:
- **Voiceover**: Centered, normalized to -14 LUFS, crisp highs with noise reduction.
- **Background Music**: Subtle ambient corporate / industrial synth track (e.g. tech/minimalist drone), ducked to -26 dB during speech and -18 dB during intro/outro.

---

## Master Editing Timeline

| Scene | Time Range | Duration | Visual Source / Footage | On-Screen Overlay | Voiceover Cue | Transition |
| :-: | :-: | :-: | :--- | :--- | :--- | :--- |
| **1** | `0:00 - 0:18` | 18s | GitHub repository hero view (`README.md` title + badges) | Title Card: **InspectAI**<br>Subtitle: *Industrial Machine Health Monitoring* | *"Hello! My name is [Name], and this is InspectAI..."* | Smooth Dissolve (15 frames) |
| **2** | `0:18 - 0:35` | 17s | Static graphic: `vibration_waveform_comparison.png` | Lower-Third: **The Industrial Challenge**<br>*>40% of breakdowns caused by bearings* | *"In industrial facilities, rotating machinery operates under..."* | Slide Left (12 frames) |
| **3** | `0:35 - 0:55` | 20s | Architecture flowchart & signal cleaning plots | Lower-Third: **Signal Conditioning Pipeline**<br>*Polyphase 12 kHz • 4th-Order Bandpass* | *"InspectAI benchmarks on authentic accelerometer recordings..."* | Cross-Zoom (10 frames) |
| **4** | `0:55 - 1:15` | 20s | Schematic: `cnn1d_architecture.png` | Lower-Third: **Edge-Optimized 1D CNN**<br>*73k Parameters • Sub-4 ms CPU Inference* | *"Rather than converting waveforms into heavy 2D spectrograms..."* | Hard Cut to Screen |
| **5** | `1:15 - 1:38` | 23s | Screen recording: Streamlit dashboard replaying `118.mat` at 1.0x/2.0x | Lower-Third: **Simulated Live Monitoring**<br>*WebSocket Telemetry • 42 ms Step Rate* | *"Here is our Streamlit monitoring console..."* | Smooth Pan Right |
| **6** | `1:38 - 1:58` | 20s | Screen recording: KPI cards & horizontal probability bars | Lower-Third: **Uncalibrated Model Scores**<br>*Standardized Confidence Labelling* | *"For each window, the CNN outputs logits across four states..."* | Zoom to Grad-CAM |
| **7** | `1:58 - 2:20` | 22s | Screen recording: Grad-CAM heatmap overlay or `gradcam_misclassifications.png` | Lower-Third: **1D Grad-CAM Explainability**<br>*Conv3 Gradients Detect BPFO Impact Spikes* | *"To provide diagnostic interpretability, 1D Grad-CAM targets..."* | Crossfade (15 frames) |
| **8** | `2:20 - 2:38` | 18s | Split screen: FastAPI Swagger (`localhost:8000/docs`) + Terminal test runner (`pytest`) | Lower-Third: **Production Hygiene & CI**<br>*Strict Recording Split • 140 Passing Tests* | *"Behind the scenes, we enforce a strict recording-level split..."* | Wipe Down (12 frames) |
| **9** | `2:38 - 2:53` | 15s | Static graphic: `model_evaluation_summary.png` | Callout Box: **Holdout Test Evaluation**<br>*39.73% Accuracy • Domain Shift Disclosed* | *"In evaluating on four untouched holdout recordings..."* | Smooth Dissolve (18 frames) |
| **10**| `2:53 - 3:00` | 7s | Closing Card: GitHub repository link & author contact info | Outro Card: **InspectAI on GitHub**<br>*github.com/[username]/inspectai* | *"InspectAI showcases end-to-end ML engineering..."* | Fade to Black (24 frames) |

---

## Detailed Editing Notes by Scene

### Scene 1: Introduction (0:00 - 0:18)
- **Footage**: Clean slow-scroll capture of the GitHub repository page or static hero slide with dark theme (`#0F172A`).
- **Text Overlay**: `INSPECTAI` in large bold font (`#38BDF8`), with presenter name and subtitle.
- **Audio Cue**: Music starts at -18 dB, ducks to -26 dB as voiceover starts at 0:02.

### Scene 2: The Industrial Challenge (0:18 - 0:35)
- **Footage**: `vibration_waveform_comparison.png` filling the screen.
- **Animation**: Gentle pan across the normal waveform to the shock transients of ball and outer race faults.
- **Lower-Third**: Show problem statistic: "Bearing Failures = >40% of Rotating Machinery Breakdowns".

### Scene 3: Preprocessing & Architecture (0:35 - 0:55)
- **Footage**: Clean presentation of the Mermaid diagram from README, highlighting the flow from raw CWRU files through `SignalCleaner` and `WindowSegmenter`.
- **Text Overlay**: "12 kHz Polyphase Decimation • Zero-Phase Butterworth Bandpass (20 - 5,000 Hz)".

### Scene 4: 1D CNN Architecture (0:55 - 1:15)
- **Footage**: High-resolution graphic of `cnn1d_architecture.png`.
- **Motion**: Subtle zoom-in highlighting the kernel sizes ($K=15 \to 7 \to 3$) and the compact parameter count ($73,156$).
- **Callout**: "Inference Latency: ~3 ms on CPU".

### Scene 5: Streamlit Live Monitoring (1:15 - 1:38)
- **Footage**: 1080p OBS screen recording of the Streamlit dashboard (`http://localhost:8501`).
- **Action Shown**: Presenter selects `118.mat (Ball Fault)` and hits **Start Simulation**. The dynamic waveform chart updates sequentially at the simulated 42 ms interval.
- **Visual Cue**: Red/amber status indicator showing active streaming.

### Scene 6: Telemetry & Probabilities (1:38 - 1:58)
- **Footage**: Close-up crop of the Streamlit KPI cards and horizontal softmax probability bars.
- **Action Shown**: High uncalibrated probability for Ball Fault (>99%).
- **Lower-Third**: "Notice: Uncalibrated Softmax Scores (Not Empirical Probabilities)".

### Scene 7: 1D Grad-CAM Explainability (1:58 - 2:20)
- **Footage**: OBS screen capture of the Grad-CAM interactive section or high-res overlay of `gradcam_misclassifications.png`.
- **Animation**: Highlight the 21 periodic ground-truth impact spikes along the 2,048-sample window aligning with BPFO (~105 Hz).
- **Lower-Third**: "Conv3 Feature Gradients Align with Outer Race Impact Frequency (BPFO ~105 Hz)".

### Scene 8: Production Engineering & Automated Tests (2:20 - 2:38)
- **Footage**: Split screen: Left half shows interactive Swagger UI (`http://localhost:8000/docs`); right half shows PowerShell terminal running `python -m pytest tests/` with green pass lines.
- **Action Shown**: Terminal outputs `140 passed in 12s`.

### Scene 9: Transparent Evaluation & Limitations (2:38 - 2:53)
- **Footage**: Full-screen display of `model_evaluation_summary.png`.
- **Visual Focus**: The bar chart showing 88.58% CV vs 39.73% holdout test, and the red 0.00% callout for Inner/Outer race faults.
- **Lower-Third / Warning Banner**: "Scientific Disclosure: Intermediate Defect Domain Shift Requires Domain Adaptation".

### Scene 10: Conclusion & Outro (2:53 - 3:00)
- **Footage**: Dark closing title card with GitHub URL and author portfolio link.
- **Audio Cue**: Voiceover finishes at 2:58; background music swells smoothly to -18 dB before fading out to silence at 3:00.


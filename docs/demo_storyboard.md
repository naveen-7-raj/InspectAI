# InspectAI: Video Demonstration Storyboard

**Total Estimated Duration**: 2 minutes 50 seconds  
**Format**: 1080p 60fps screen recording with voiceover narration  
**Objective**: Clear, recruiter-focused video showcase of InspectAI's ML engineering, signal processing, explainability, and software architecture.

---

## Storyboard Overview Table

| Scene # | Duration | Scene Title | Visual Screen Content | Required Asset | Transition |
| :-: | :-: | :--- | :--- | :--- | :--- |
| **1** | 20s | Introduction | GitHub repo header & title badges | `README.md` top hero view | Fade to Slide |
| **2** | 20s | Problem Statement | 4-class waveform comparison plots | `docs/images/vibration_waveform_comparison.png` | Slide Left |
| **3** | 25s | System Architecture | Mermaid system flow & 1D CNN layer diagram | `docs/images/cnn1d_architecture.png` | Cut to Screen |
| **4** | 25s | Dashboard & Live Stream | Streamlit UI with real-time waveform chart | `docs/images/demo/02_simulated_streaming_active.png` | Zoom to KPIs |
| **5** | 25s | Prediction & Scores | Live KPI cards & uncalibrated softmax distribution | `docs/images/demo/03_prediction_results_probabilities.png` | Pan Down |
| **6** | 25s | 1D Grad-CAM Explainability | Saliency overlay & periodic impact spikes (BPFO) | `docs/images/gradcam_misclassifications.png` | Cut to Terminal |
| **7** | 20s | Engineering & CI | FastAPI Swagger docs & 140 passing pytest tests | `http://localhost:8000/docs` & `pytest` CLI | Wipe Right |
| **8** | 15s | Scientific Limitations | Holdout test confusion matrix & domain shift analysis | `docs/images/final_test_confusion_matrix.png` | Dissolve |
| **9** | 10s | Conclusion & Links | GitHub repo URL & contact info | Presentation closing slide | Fade to Black |

---

## Detailed Scene Specifications

### Scene 1: Project Overview & Intro (0:00 – 0:20)
- **Duration**: 20 seconds
- **Screen Content**: Clean browser display of the InspectAI GitHub repository showing title, technology badges (PyTorch, FastAPI, Streamlit, 140 Tests Passing), and introduction.
- **Narration**: *"Hello! My name is [Name], and this is InspectAI—an explainable machine learning engineering platform for industrial machine health monitoring..."*
- **Required Asset**: Live browser on GitHub repo or `README.md`.
- **Action**: Smooth scroll down past the problem statement to the badges.
- **Transition**: Smooth fade to Scene 2.

### Scene 2: The Industrial Challenge (0:20 – 0:40)
- **Duration**: 20 seconds
- **Screen Content**: High-resolution waveform comparison plot highlighting baseline vibrations versus impulsive ball, inner race, and outer race shocks.
- **Narration**: *"In manufacturing plants, bearing failures cause over 40% of motor breakdowns. Standard deep learning models often leak overlapping data and act as opaque black boxes..."*
- **Required Asset**: `docs/images/vibration_waveform_comparison.png`.
- **Action**: Highlight the distinct impact spikes of inner and outer race defects.
- **Transition**: Slide left to Scene 3.

### Scene 3: Modular Architecture & Signal Conditioning (0:40 – 1:05)
- **Duration**: 25 seconds
- **Screen Content**: Architecture flowchart showing polyphase resampling to 12 kHz, DC offset removal, 4th-order Butterworth bandpass filtering, and the 73,156-parameter 1D CNN structure.
- **Narration**: *"InspectAI implements an end-to-end pipeline: polyphase decimation to 12 kHz, zero-phase bandpass filtering, and a 1D CNN with just 73k parameters optimized for sub-4-millisecond CPU inference..."*
- **Required Asset**: `docs/images/cnn1d_architecture.png`.
- **Action**: Pan across Conv Block 1, Block 2, and Block 3 to the classification head.
- **Transition**: Cut directly to live application in Scene 4.

### Scene 4: Streamlit Dashboard & WebSocket Telemetry (1:05 – 1:30)
- **Duration**: 25 seconds
- **Screen Content**: Streamlit console at `http://localhost:8501`. Presenter selects recording `118.mat` (Ball Fault) and clicks "Start Replay".
- **Narration**: *"Here is our Streamlit monitoring dashboard. To simulate continuous sensor monitoring without physical hardware, InspectAI features a WebSocket streaming gateway that replays pre-recorded CWRU signals chunk-by-chunk..."*
- **Required Asset**: Live Streamlit UI or `docs/images/demo/02_simulated_streaming_active.png`.
- **Action**: Show live waveform plot dynamically refreshing at the simulated step rate.
- **Transition**: Zoom in on prediction metrics.

### Scene 5: Real-Time Predictions & Uncertainty Awareness (1:30 – 1:55)
- **Duration**: 25 seconds
- **Screen Content**: KPI cards showing processed windows, anomaly counters, and horizontal probability distribution bars.
- **Narration**: *"For each 2,048-sample window, the CNN outputs logits across four health states. We explicitly emphasize that softmax outputs are uncalibrated model scores rather than true empirical probabilities..."*
- **Required Asset**: `docs/images/demo/03_prediction_results_probabilities.png`.
- **Action**: Cursor hovers over the uncalibrated confidence notice and inference latency timer (~3 ms).
- **Transition**: Pan down to the Grad-CAM explanation section.

### Scene 6: 1D Grad-CAM Explainability (1:55 – 2:20)
- **Duration**: 25 seconds
- **Screen Content**: Grad-CAM saliency heatmap overlaid across the vibration window, contrasting the flat predicted Ball Fault attribution against the 21 periodic ground-truth impact spikes.
- **Narration**: *"To provide mechanical interpretability, 1D Grad-CAM targets our penultimate convolutional layer (`conv3`). On unseen outer race faults, Grad-CAM detected 21 periodic spikes matching the 105 Hz Ball Pass Frequency..."*
- **Required Asset**: `docs/images/gradcam_misclassifications.png`.
- **Action**: Highlight the periodic temporal intervals where attribution exceeds 0.60.
- **Transition**: Cut to terminal and API documentation.

### Scene 7: Production Engineering, API & Test Suite (2:20 – 2:40)
- **Duration**: 20 seconds
- **Screen Content**: Split screen: FastAPI Swagger interactive documentation (`/docs`) on the left, and terminal running `pytest tests/` passing 140 tests on the right.
- **Narration**: *"Behind the scenes, InspectAI is built with production hygiene: Pydantic v2 schemas, OpenAPI 3.1 documentation, strict recording-level isolation, and 140 automated tests with 100% pass rate..."*
- **Required Asset**: `http://localhost:8000/docs` and terminal showing `140 passed in 12s`.
- **Action**: Execute `pytest tests/ -v` or showcase the green test suite.
- **Transition**: Dissolve to evaluation results.

### Scene 8: Transparent Evaluation & Failure Mode Analysis (2:40 – 2:55)
- **Duration**: 15 seconds
- **Screen Content**: Holdout test confusion matrix showing 39.73% overall accuracy (0% on unseen 0.014" race faults).
- **Narration**: *"True engineering requires scientific honesty. While our model achieved 88.6% cross-validation accuracy, it collapsed to 39.7% on unseen holdout defects due to intermediate defect geometry and load-zone domain shift..."*
- **Required Asset**: `docs/images/final_test_confusion_matrix.png`.
- **Action**: Highlight the diagonal for Normal and Ball Fault, and the off-diagonal default for Inner/Outer race faults.
- **Transition**: Dissolve to conclusion slide.

### Scene 9: Conclusion & Portfolio Wrap-up (2:55 – 3:05)
- **Duration**: 10 seconds
- **Screen Content**: Final slide displaying GitHub repository link, author contact details, and key project takeaways.
- **Narration**: *"InspectAI demonstrates practical ML engineering from signal conditioning to model interpretability. Thank you for watching!"*
- **Required Asset**: Repository summary slide or GitHub homepage.
- **Action**: End voiceover and display contact info.
- **Transition**: Fade to black.


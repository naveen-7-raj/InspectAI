# InspectAI: Polished Video Demonstration Narration Script

**Total Spoken Word Count**: 395 words (~2 minutes 45 seconds at 140 WPM)  
**Target Duration**: 2:30 to 3:00  
**Target Audience**: AI/ML Engineer Intern Recruiters & Machine Learning Engineers  
**Tone**: Technical, clear, humble, and scientifically transparent.

---

### [0:00 - 0:18] Scene 1: Introduction
**Visual**: Repository title, clean architecture diagram, and GitHub README header.  
**Spoken Script**:  
> "Hello! My name is [Your Name], and this is **InspectAI**, an explainable machine learning platform engineered for industrial rotational machine health monitoring. 
> InspectAI diagnoses bearing defects from high-frequency accelerometer vibrations using a lightweight 1D CNN, 1D Grad-CAM interpretability, and real-time streaming telemetry."

---

### [0:18 - 0:35] Scene 2: The Industrial Bearing-Monitoring Problem
**Visual**: 4-class vibration waveform comparison (`docs/images/vibration_waveform_comparison.png`).  
**Spoken Script**:  
> "In industrial facilities, rotating machinery operates under continuous mechanical stress. Rolling element bearing failures cause over 40% of all electric motor breakdowns, costing plants billions in unplanned downtime. 
> While accelerometers detect microscopic impact shocks, deep learning models often leak overlapping temporal data and act as opaque black boxes that plant engineers cannot verify."

---

### [0:35 - 0:55] Scene 3: Preprocessing & Modular System Architecture
**Visual**: Pipeline flowchart and signal conditioning diagrams.  
**Spoken Script**:  
> "InspectAI benchmarks on authentic accelerometer recordings from the Case Western Reserve University Bearing Data Center. 
> Our preprocessing pipeline resamples signals to 12 kHz using polyphase anti-aliasing, removes DC sensor bias, and applies a 4th-order zero-phase Butterworth bandpass filter from 20 Hz to 5,000 Hz. 
> We then segment signals into 2,048-sample sliding windows with 75% overlap."

---

### [0:55 - 1:15] Scene 4: Lightweight 1D CNN & Inference Efficiency
**Visual**: 1D CNN layer diagram (`docs/images/cnn1d_architecture.png`).  
**Spoken Script**:  
> "Rather than converting waveforms into heavy 2D spectrograms, our classifier processes raw 1D time-series directly. 
> The architecture features three Conv1D blocks with batch normalization, max pooling, adaptive spatial pooling, and dropout. 
> With just 73,156 trainable parameters, it executes on CPU in under 4 milliseconds per window—ideal for low-power edge gateways."

---

### [1:15 - 1:38] Scene 5: Streamlit Dashboard & WebSocket Telemetry
**Visual**: Streamlit UI with live simulated waveform updating on recording `118.mat`.  
**Spoken Script**:  
> "Here is our Streamlit monitoring console. 
> To simulate continuous sensor monitoring without physical motor hardware, InspectAI features a WebSocket streaming gateway that replays pre-recorded CWRU signals chunk-by-chunk at authentic 42-millisecond steps. 
> The console plots the live dynamic waveform, tracks inference latency, and displays real-time health classifications."

---

### [1:38 - 1:58] Scene 6: Predictions & Uncertainty Awareness
**Visual**: Live telemetry KPI cards and horizontal softmax distribution bars.  
**Spoken Script**:  
> "For each window, the CNN outputs logits across four states: Normal, Ball Fault, Inner Race Fault, and Outer Race Fault. 
> On this Ball Fault recording, the model accurately flags the defect. 
> Importantly, we follow reliable ML engineering standards by explicitly labeling these outputs as uncalibrated neural network scores rather than true empirical probabilities."

---

### [1:58 - 2:20] Scene 7: 1D Grad-CAM Explainability
**Visual**: Saliency overlay plot showing impact peaks (`docs/images/gradcam_misclassifications.png`).  
**Spoken Script**:  
> "To provide diagnostic interpretability, 1D Grad-CAM targets our penultimate convolutional layer (`conv3`). 
> When diagnosing an outer race fault, Grad-CAM reveals 21 sharp periodic impact spikes matching the exact Ball Pass Frequency of the Outer Race (BPFO around 105 Hz). 
> This proves the convolutional filters captured physical impact acoustics, isolating model attribution from the classification head's decision."

---

### [2:20 - 2:38] Scene 8: Recording-Level Splitting & Production Engineering
**Visual**: Recording partition diagram (`docs/images/recording_level_split.png`) and Swagger `/docs`.  
**Spoken Script**:  
> "Behind the scenes, we enforce a strict recording-level split: windows from the same recording are never split across train and test sets, eliminating data leakage. 
> The system is packaged with a modular FastAPI backend, OpenAPI 3.1 documentation, and 140 automated unit and integration tests passing with 100% success on CPU."

---

### [2:38 - 2:53] Scene 9: Transparent Evaluation Results & Honest Limitations
**Visual**: Holdout confusion matrix (`docs/images/final_test_confusion_matrix.png`).  
**Spoken Script**:  
> "In evaluating on four untouched holdout recordings, the model achieved 39.73% accuracy and 34.04% Macro-F1. 
> While it achieved 90.3% F1 on Normal bearings and 45.9% on Ball Faults, it scored 0% on unseen 0.014-inch race faults due to defect geometry and load-zone domain shift. 
> The model is not production-ready and requires domain adaptation before industrial deployment."

---

### [2:53 - 3:00] Scene 10: Conclusion
**Visual**: Repository summary slide and GitHub link.  
**Spoken Script**:  
> "InspectAI showcases end-to-end ML engineering: rigorous signal processing, leakage-safe evaluation, model interpretability, and full-stack software delivery. 
> Thank you for watching!"

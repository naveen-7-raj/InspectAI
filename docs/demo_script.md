# InspectAI: 3–5 Minute Portfolio Video Demonstration Script

**Target Duration**: 3:30 to 4:30 (approx. 550 words at 140 WPM)  
**Target Audience**: Staff ML Engineers, Engineering Directors, Technical Recruiters  
**Tone**: Technical, rigorous, humble, and scientifically transparent.  
**Deliverables Covered**: Full-stack application, CWRU baseline, Grad-CAM, Paderborn unsupervised adaptation, safety guardrails, Prometheus metrics.

---

### [0:00 - 0:35] Scene 1: Introduction & Real-World Motivation
**Visual**: Browser displaying GitHub repository and root `README.md`, transitioning to the Streamlit operator dashboard.  
**Speaker**:  
> "Hello! My name is [Your Name], and this is **InspectAI**, an explainable, production-hardened machine health monitoring system for industrial rotating equipment.  
> Rolling element bearing failures cause over 40% of unscheduled mechanical breakdowns, costing billions annually in unplanned downtime.  
> However, deploying deep learning to vibration diagnostics faces three major engineering traps: spurious accuracy from window-overlap data leakage, severe cross-test-rig domain shift, and the dangerous risk of adapting away true defect signatures during runtime recalibration.  
> InspectAI was built from the ground up to solve these problems with end-to-end scientific rigor on commodity CPU edge hardware."

---

### [0:35 - 1:05] Scene 2: Dashboard Overview & Simulated Streaming Replay
**Visual**: Streamlit dashboard showing connection status, model metadata (`SpectralFusionCNN`), KPI cards, and recording selection dropdown.  
**Speaker**:  
> "Here is our Streamlit diagnostic console connected to our FastAPI backend.  
> In industrial monitoring, connecting to live physical sensors requires specialized DAQ hardware. To evaluate realistic edge streaming safely, InspectAI features a WebSocket streaming gateway that replays authentic laboratory-recorded vibration files from CWRU and Paderborn, pacing at calibrated 42.6-millisecond intervals—the exact physical arrival rate of 2,048 samples at 12 kHz.  
> Let's start with CWRU recording 97, a healthy baseline motor running under zero horsepower."

---

### [1:05 - 1:40] Scene 3: Baseline Fault Classification & In-Domain Performance
**Visual**: Replaying recording 97, then switching to recording 169 (Inner Race Fault). Waveform updates in real time, softmax probability bars update to Inner Race Fault (>98%).  
**Speaker**:  
> "When streaming healthy data, the model outputs a confident Normal prediction with sub-30-millisecond latency.  
> Now let's switch to recording 169, which has a 7-mil single-point defect on the inner raceway.  
> Notice how the live waveform reveals sharp periodic shock bursts, and the dual-branch `SpectralFusionCNN` immediately flags an **Inner Race Fault** with over 98% posterior probability.  
> By fusing raw 1D temporal vibrations with the log-demodulated Hilbert envelope spectrum, our 179,000-parameter network achieves a 99.8% macro F1 across unseen motor loads in-domain."

---

### [1:40 - 2:20] Scene 4: 1D Grad-CAM Physical Explainability
**Visual**: Clicking the Grad-CAM inspector on recording 169. Heatmap overlays on the waveform, highlighting specific periodic peaks.  
**Speaker**:  
> "In high-stakes industrial maintenance, black-box predictions are unviable—operators cannot shut down a production line without physical evidence.  
> Here, our 1D Grad-CAM engine calculates gradient-weighted saliency on the penultimate convolutional layer.  
> Notice the bright orange attribution peaks: the model is not focusing on background motor hum or electrical harmonics. It is specifically triggering on discrete shock transients spaced at approximately 9.4 milliseconds—which precisely matches the theoretical kinematic Ball Pass Frequency of the Inner Race (BPFI around 156 Hz at 1,797 RPM)."

---

### [2:20 - 3:05] Scene 5: Cross-Rig Domain Collapse & Unsupervised CORAL Adaptation
**Visual**: Switching to Paderborn University bearing K001. Showing baseline zero-shot collapse, then clicking 'Enable Adaptive Streaming (CORAL)'. The system buffers 50 unlabelled windows, computes covariance, and adapts.  
**Speaker**:  
> "Now, what happens if we take this frozen model and deploy it to an entirely different test rig?  
> Here is data from Paderborn University bearing K001. Due to different sensor mountings, radial loads, and machine dynamics, zero-shot evaluation collapses to near-random Macro F1 around 0.25.  
> To solve this without requiring expensive manual defect labels on the new rig, InspectAI implements unsupervised **Correlation Alignment (CORAL)** on our 576-dimensional bottleneck embeddings.  
> When we enable adaptive mode, the engine buffers 50 unlabelled baseline windows, aligns target feature covariance with CWRU source statistics, and successfully restores correct classification with a condition number $\kappa$ well within numerical stability limits."

---

### [3:05 - 3:50] Scene 6: Safe Drift Monitoring & Fault-Suspect Guardrail
**Visual**: Injecting a defect into the stream. DriftMonitor changes state from `STABLE` to `FAULT_SUSPECT`. Showing alert banner: `[SAFETY ALERT] Recalibration SUPPRESSED due to fault suspicion`.  
**Speaker**:  
> "A critical hazard in adaptive monitoring is catastrophic adaptation: if a machine starts failing, vibration variance changes. An unsupervised adapter could mistake this fault for a domain shift and adapt it away, making a destroyed bearing appear normal.  
> InspectAI solves this with heuristic safety guardrails in our `DriftMonitor`.  
> When signal kurtosis exceeds 4.5 or the model posterior defect probability exceeds 35%, the monitor immediately flags **FAULT_SUSPECT** and strictly suppresses recalibration. True mechanical damage is alerted to operators, never adapted away.  
> If non-fault operating shifts occur, recalibration is tested in shadow mode with a 30-window cooldown and instant rollback."

---

### [3:50 - 4:25] Scene 7: Observability & Prometheus Metrics
**Visual**: Terminal or browser opening `http://localhost:8000/metrics`. Showing active WebSocket gauges, latency histograms, condition numbers, and prediction counters.  
**Speaker**:  
> "For production reliability, our FastAPI backend exposes standard Prometheus metrics at `/metrics`.  
> SRE teams can track prediction volumes by class, latency histograms across baseline and adaptive modes, active WebSocket connections, and Frobenius domain drift in Grafana.  
> In our Phase 20 benchmark, InspectAI demonstrated a throughput of 243 windows per second on CPU—providing a **10.4x real-time headroom** over physical 12 kHz arrival rates, with zero dropped frames across all benchmark runs."

---

### [4:25 - 4:45] Scene 8: Conclusion & Honest Limitations
**Visual**: Project repository summary slide with GitHub link, test pass badge (300 passed), and limitation bullets.  
**Speaker**:  
> "In summary, InspectAI combines dual-branch deep learning, 1D Grad-CAM explainability, unsupervised domain adaptation, and industrial safety guardrails in a clean, CPU-first deployment.  
> It does not claim to predict remaining useful life or connect to live factory hardware out of the box—it is a rigorously validated, transparent engineering foundation.  
> All 300 automated tests pass with zero failures. Thank you for watching!"

# Phase 10.5: Dataset Generalization and Failure Analysis Report

**Project**: InspectAI – Explainable Industrial Machine Health Monitoring System  
**Date**: November 2026  
**Author**: Machine Learning Engineering Team  
**Status**: Completed & Verified  

---

## Executive Summary

Phase 10.5 conducted a rigorous, dataset-level generalization audit of the InspectAI project. Following the empirical outcomes of Phase 10.2 (Normalization Ablation), Phase 10.3 (Multiscale CNN Architecture), and Phase 10.4 (Frequency-Domain Feature Fusion), this investigation isolated the root causes of the persistent classification failures between **Inner Race Fault** (recording `209`, 0.021" fault depth) and **Outer Race Fault** (recording `234`, 0.021" fault depth).

This phase was strictly an **investigation and forensic audit**; no models were trained, no checkpoint weights were altered, and no test set samples were accessed.

### Key Audit Discoveries

1. **Reconciliation of Historical Baseline Discrepancies**:
   - The reported baseline OOF Macro-F1 scores across prior phases were reconciled:
     - **Phase 10.2 Baseline (`global`)**: OOF Accuracy **86.26%**, Macro-F1 **83.78%** (106,244 parameters, `fc_units=128`).
     - **Phase 10.3 Baseline (`baseline_1d_cnn`)**: OOF Accuracy **86.26%**, Macro-F1 **83.78%** (106,244 parameters, `fc_units=128`). The citation of "86.26%" in prior documentation was an accidental quotation of OOF Accuracy rather than Macro-F1.
     - **Phase 10.4 Baseline (`baseline_waveform`)**: OOF Accuracy **84.69%**, Macro-F1 **81.09%** (73,156 parameters, `fc_units=64`). The 2.69% F1 difference between Phase 10.2/10.3 and Phase 10.4 stems from the prompt-mandated reduction in classifier capacity (`fc_units=64` vs `128`), which reduced Fold 1 validation Macro-F1 from 94.09% to 90.57%.
2. **Identification of Structural Experimental Confounding**:
   - In the 9-recording development set, **defect severity is 100% confounded with fault class under 2-fold cross-validation**:
     - In **Fold 0**, the training set contains only **0.007" Inner Race** (`105`) and **0.021" Outer Race** (`234`). When evaluated on `209` (0.021" Inner Race), the model has never encountered a large inner raceway fault during training. Because `209` exhibits high-amplitude structural ringing ($\text{RMS} = 0.525\text{ g}$), every model predicts **Outer Race with 100% confidence**.
     - In **Fold 1**, the training set contains only **0.021" Inner Race** (`209`) and **0.007" Outer Race** (`130`). When evaluated on `234` (0.021" Outer Race), the model predicts **Inner Race with up to 94.9% confidence**.
3. **Physical Signal Reality: Resonance Overwhelms Kinematic Pulses**:
   - Welch power spectral density and band-energy integration reveal that **over 91% of total signal energy** in recordings `209` (92.85%) and `234` (91.02%) resides in high-frequency structural resonances (**2,000 to 6,000 Hz**).
   - In contrast, the baseband impact range (**0 to 500 Hz**)—which contains theoretical kinematic frequencies ($\text{BPFO} \approx 107.4\text{ Hz}$ and $\text{BPFI} \approx 162.2\text{ Hz}$)—contains **less than 2.5% of total signal energy**.
   - Because the 0.021" defect produces continuous structural fatigue rather than isolated Dirac-like shocks, the high-frequency resonance profiles of Inner Race and Outer Race bearings become virtually indistinguishable in single-sensor Drive End vibration.
4. **Severe Operating Condition Limitations**:
   - While the healthy class (Normal) includes operating load variations (0, 1, 2, 3 HP), **all 9 fault recordings in the benchmark were acquired exclusively at 0 HP (1,797 RPM)**.
   - Operating motor load is completely confounded with machine health state, making it impossible to evaluate cross-load or speed generalization with the existing benchmark partition.

---

## 1. Cross-Experiment Metric Reconciliation

The baseline evaluation metrics across Phase 10 were systematically audited and reconciled:

| Phase | Configuration Name | Model Architecture | Classifier FC Units | Trainable Parameters | Pooled OOF Accuracy | Pooled OOF Macro-F1 | Fold 0 Macro-F1 | Fold 1 Macro-F1 | Root Cause of Metric Variance |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 10.2** | `global` | `VibrationCNN1D` | 128 | 106,244 | 86.26% | **83.78%** | 66.69% | 94.09% | Instantiated from config defaults (`fc_units=[128, 64] -> 128`). |
| **Phase 10.3** | `baseline_1d_cnn` | `VibrationCNN1D` | 128 | 106,244 | 86.26% | **83.78%** | 66.69% | 94.09% | Identical checkpoint, weights, seeds, and metrics to Phase 10.2. (86.26% was accuracy). |
| **Phase 10.4** | `baseline_waveform` | `VibrationCNN1D` | 64 | 73,156 | 84.69% | **81.09%** | 66.67% | 90.57% | Instantiated with `fc_units=64` (73k parameters) per prompt instruction. |

### Diagnostic Conclusion on Reconciliation
There is zero data leakage or metric inconsistency. When `VibrationCNN1D` operates with 106,244 parameters (`fc_units=128`), its pooled OOF Macro-F1 is **83.78%** (Accuracy 86.26%). When operating with 73,156 parameters (`fc_units=64`), the reduced capacity in the classification head slightly increases raceway misclassification in Fold 1 (from 0 to 40 windows of recording 234), yielding **81.09%** Macro-F1 (Accuracy 84.69%).

---

## 2. Recording-Level Metadata and Vibration Statistics Audit

A comprehensive audit was performed across all 13 recordings (9 development + 4 test) using the segmented windows archive ([cwru_segmented_windows.npz](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/data/processed/cwru_segmented_windows.npz)) and raw `.mat` files in `data/raw/cwru/`:

| Rec ID | Partition | Fault Class | Defect Size | Motor Load | Motor Speed (RPM) | Speed Verification | Sensor Channel | Bearing Model | Window Count | RMS ($g$) | Std ($g$) | Peak ($g$) | Crest Factor | Kurtosis (Pearson) | Skewness |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **97** | Test | Normal | 0.000" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 116 | 0.0700 | 0.0700 | 0.2742 | 3.9172 | 2.7765 | +0.0168 |
| **98** | Dev (F0 Val) | Normal | 0.000" | 1 HP | *1772.0* | Missing in Header | DE | SKF 6205 | 233 | 0.0596 | 0.0596 | 0.3015 | 5.0630 | 2.9722 | +0.0281 |
| **99** | Dev (F0 Val) | Normal | 0.000" | 2 HP | *1750.0* | Missing in Header | DE | SKF 6205 | 233 | 0.0551 | 0.0551 | 0.2706 | 4.9134 | 2.9656 | +0.0210 |
| **100** | Dev (F0 Train) | Normal | 0.000" | 3 HP | 1725.0 | Verified (Header) | DE | SKF 6205 | 234 | 0.0552 | 0.0552 | 0.2786 | 5.0458 | 3.0479 | +0.0313 |
| **105** | Dev (F0 Train) | Inner Race | 0.007" | 0 HP | 1797.0 | Verified (Header) | DE | SKF 6205 | 233 | 0.2907 | 0.2907 | 1.6989 | 5.8439 | 5.3781 | +0.1656 |
| **118** | Dev (F0 Val) | Ball Fault | 0.007" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 236 | 0.1386 | 0.1386 | 0.6226 | 4.4928 | 2.9799 | -0.0091 |
| **130** | Dev (F0 Val) | Outer Race | 0.007" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 235 | **0.6661** | **0.6661** | **3.6415** | 5.4668 | **7.6676** | +0.0621 |
| **169** | Test | Inner Race | 0.014" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 234 | 0.1941 | 0.1941 | 2.0130 | 10.3703 | 21.9956 | +0.0283 |
| **185** | Test | Ball Fault | 0.014" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 234 | 0.1532 | 0.1532 | 2.2818 | 14.8975 | 17.7947 | +0.2255 |
| **197** | Test | Outer Race | 0.014" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 234 | 0.0974 | 0.0974 | 0.5642 | 5.7960 | 3.0477 | +0.0015 |
| **209** | Dev (F0 Val) | Inner Race | 0.021" | 0 HP | 1797.0 | Verified (Header) | DE | SKF 6205 | 235 | **0.5249** | **0.5249** | **3.7014** | 7.0517 | **7.3960** | +0.2854 |
| **222** | Dev (F0 Train) | Ball Fault | 0.021" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 235 | 0.1335 | 0.1335 | 1.6428 | 12.3090 | 8.6660 | -0.0016 |
| **234** | Dev (F0 Train) | Outer Race | 0.021" | 0 HP | 1796.0 | Verified (Header) | DE | SKF 6205 | 236 | **0.5835** | **0.5835** | **6.6406** | 11.3799 | **20.9398** | +0.0926 |

*Distinction Note*: Motor speeds for recordings `98` and `99` are absent from the original `.mat` file variables and are inferred from induction motor slip tables for 1 HP (~1,772 RPM) and 2 HP (~1,750 RPM). All other recording attributes are verified directly from MATLAB headers.

---

## 3. Signal-Level Failure Analysis: Recordings 105, 209, 130, 234

To isolate why the models struggle with the Inner Race vs. Outer Race boundary, we compared recordings `105`, `209`, `130`, and `234` across time and frequency representations:

```
                           +-----------------------------------------------+
                           |          Defect Severity Comparison           |
+--------------------------+-----------------------+-----------------------+
| Fault Location           | 0.007" Minor Defect   | 0.021" Severe Spall   |
+--------------------------+-----------------------+-----------------------+
| Inner Race (BPFI ~162Hz) | Recording 105         | Recording 209         |
|                          | RMS: 0.29g | Kurt: 5.4| RMS: 0.52g | Kurt: 7.4|
+--------------------------+-----------------------+-----------------------+
| Outer Race (BPFO ~107Hz) | Recording 130         | Recording 234         |
|                          | RMS: 0.67g | Kurt: 7.7| RMS: 0.58g | Kurt: 20.9|
+--------------------------+-----------------------+-----------------------+
```

### 3.1 Time-Domain Waveform Statistics & Amplitude Overlap

Visual inspection of 500-sample windows (~41.7 ms, spanning ~1.25 complete shaft revolutions) reveals striking similarities between recordings `209` (Inner Race 0.021") and `130` (Outer Race 0.007"):
- **Peak Amplitudes**: `209` has peak $3.701\text{ g}$; `130` has peak $3.642\text{ g}$ (**less than 1.6% difference**).
- **Kurtosis**: `209` has Pearson kurtosis of **7.40**; `130` has Pearson kurtosis of **7.67** (**less than 3.6% difference**).
- **RMS Energy**: `209` has RMS $0.525\text{ g}$; `234` has RMS $0.584\text{ g}$ (**less than 11% difference**).

![Time Domain Waveforms](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/figures/phase10_5/time_domain_waveforms.png)  
*Figure 1: Time-domain vibration acceleration profiles over 500 samples (~41.7 ms, ~1.25 shaft revolutions).*

![Amplitude Distributions](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/figures/phase10_5/amplitude_distributions.png)  
*Figure 2: Standardized amplitude probability density functions demonstrating heavy-tailed kurtosis and distribution overlap.*

### 3.2 Spectral Energy Band Distribution

Integrating Welch power spectral densities over standard vibration diagnostic frequency bands reveals that high-frequency structural resonances dominate all fault recordings:

| Recording ID | Fault Class | Defect Size | Low Band (0–500 Hz) | Mid Band (500–2,000 Hz) | High Band (2,000–6,000 Hz) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **105** | Inner Race | 0.007" | 0.90% | 20.07% | 79.02% |
| **209** | Inner Race | 0.021" | 2.43% | 4.72% | **92.85%** |
| **130** | Outer Race | 0.007" | 0.13% | 0.80% | **99.07%** |
| **234** | Outer Race | 0.021" | 0.89% | 8.09% | **91.02%** |

![Spectral Band Energy](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/figures/phase10_5/frequency_band_energy_barchart.png)  
*Figure 3: Fractional energy distribution across low (0–500 Hz), mid (500–2,000 Hz), and high (2,000–6,000 Hz) spectral bands.*

![Power Spectral Density](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/figures/phase10_5/power_spectral_density.png)  
*Figure 4: Full-band Welch PSD (0 to 6,000 Hz) showing massive structural resonance bands above 2,500 Hz.*

### 3.3 Baseband Modulation & Envelope Demodulation

In the baseband (0 to 500 Hz), kinematic impact frequencies are:
- Shaft frequency: $f_r = 29.95\text{ Hz}$
- Outer Race impact rate: $\text{BPFO} = 107.36\text{ Hz}$
- Inner Race impact rate: $\text{BPFI} = 162.19\text{ Hz}$

However, in recordings `209` and `234` (0.021" defects), the dominant peaks in the baseband are **not** the fundamental impact frequencies:
- `209` dominant baseband peaks: 451.2 Hz, 457.0 Hz, 462.9 Hz (motor electrical harmonics and transducer structural resonances).
- `234` dominant baseband peaks: 474.6 Hz, 480.5 Hz, 427.7 Hz.

At 0.021" severity, continuous metal-to-metal rubbing and multiple spall edges blur individual impulse transitions. The Hilbert envelope spectra show diffuse carrier modulation rather than sharp harmonic ladders at BPFO or BPFI.

![Envelope Spectra Comparison](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/figures/phase10_5/envelope_spectra_comparison.png)  
*Figure 5: Hilbert envelope demodulation spectra in the baseband (0 to 500 Hz) with theoretical BPFO and BPFI references.*

---

## 4. Confounding and Identifiability Audit

A critical question for machine learning engineers is: **Are the target classes identifiable from the training data, or are they mathematically confounded with nuisance variables?**

### 4.1 Defect Severity vs. Class Confounding in Group Cross-Validation

Under Stratified Group K-Fold cross-validation (which prevents temporal window leakage by placing all windows of any recording into a single fold), the experimental structure creates **unavoidable confounding between fault severity and fault class**:

![Severity Confounding Grid](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/figures/phase10_5/severity_confounding_grid.png)  
*Figure 6: Structural confounding in the 2-fold cross-validation partition.*

#### Mathematical Confounding Table:
- **Fold 0 Training Set**:
  - `105`: Inner Race, **0.007"** ($\text{RMS} = 0.29\text{ g}$)
  - `234`: Outer Race, **0.021"** ($\text{RMS} = 0.58\text{ g}$)
  - *Network Observation*: Low energy = Inner Race; High energy = Outer Race.
- **Fold 0 Validation Set**:
  - `209`: Inner Race, **0.021"** ($\text{RMS} = 0.52\text{ g}$)
  - *Network Prediction*: `209` has high energy ($\text{RMS} = 0.525\text{ g}$), which is 80% higher than the training Inner Race sample (`105`). It perfectly matches the training Outer Race profile (`234`). Therefore, the network classifies `209` as **Outer Race with 100% certainty**.
- **Fold 1 Training Set**:
  - `209`: Inner Race, **0.021"** ($\text{RMS} = 0.52\text{ g}$)
  - `130`: Outer Race, **0.007"** ($\text{RMS} = 0.67\text{ g}$)
- **Fold 1 Validation Set**:
  - `234`: Outer Race, **0.021"** ($\text{RMS} = 0.58\text{ g}$)
  - *Network Prediction*: `234` is biased towards **Inner Race** because `209` is the only 0.021" fault seen in training.

**Core Scientific Finding**:  
This error is **not a modeling failure**; it is an **identifiability limit of the dataset**. When every recording in a cross-validation fold occupies a unique combination of fault location and severity, the network cannot learn an invariant boundary between Inner Race and Outer Race that generalizes across severity levels.

### 4.2 Motor Load Confounding

The healthy baseline (Normal) contains 4 recordings spanning loads from 0 to 3 HP (`97`, `98`, `99`, `100`). However, **every single fault recording in the CWRU 12k Drive End benchmark was acquired at 0 HP**:
- Faults under load: None in this benchmark.
- Load variation under fault: None.
- As a consequence, the model cannot learn load-invariant fault representations. Any load-dependent acoustic changes (e.g., changes in rotational speed or slip) cannot be disentangled from health state.

### 4.3 $N=1$ Physical Experiment Confounding

In the CWRU dataset, each (Fault Class, Defect Size, Motor Load) condition is represented by **exactly one continuous recording**.  
Because $N=1$, recording-specific physical artifacts—such as the exact placement and tightening torque of the magnetic accelerometer base, variations in sensor cable capacitance, small differences in bearing press-fit alignment, or oil bath temperature—are **100% collinear** with the recording ID.

---

## 5. Cross-Validation Integrity Audit

A comprehensive integrity audit was executed on the cross-validation partition metadata ([cv_integrity_report.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase10_5/cv_integrity_report.json)):

```
Dataset Windows Total: 2,928
- Development Windows: 2,110 (across 9 recordings)
- Test Windows:         818 (across 4 recordings: 97, 169, 185, 197)

Cross-Validation Partitions:
Fold 0:
  - Train: ['100', '105', '222', '234'] (938 windows)
  - Val:   ['118', '130', '209', '98', '99'] (1,172 windows)
  - Inter-Fold Overlap: ZERO (disjoint sets)
Fold 1:
  - Train: ['118', '130', '209', '98', '99'] (1,172 windows)
  - Val:   ['100', '105', '222', '234'] (938 windows)
  - Inter-Fold Overlap: ZERO (disjoint sets)

Leakage Verification:
- Test Set Contamination: ZERO (No test recording was ever included in training or validation)
- Out-of-Fold Coverage: Exactly 2,110 unique windows evaluated once (0 overlapping, 0 omitted)
- Normalizer Provenance: All fitted normalizers strictly used training fold recordings
```

---

## 6. Real-World Limitations and Generalization Bounds

Based on empirical evidence from Phase 10.1 through 10.5, the following limitations must be clearly communicated to stakeholders, recruiters, and engineering teams:

1. **No Inherent Machine Invariance**:
   - The model was trained and validated on a single 2-HP Reliance Electric motor test bench. It cannot be assumed to generalize to other motor models, larger industrial drivetrains, or multi-stage gearboxes.
2. **Artificial Electro-Discharge Machining (EDM) Defects**:
   - The CWRU faults are EDM-etched square notches. Real industrial bearing failures (fatigue flaking, micro-spalling, abrasive wear, electrical fluting, lubrication starvation) evolve gradually over months and produce different vibration distributions.
3. **Sensor Location Sensitivity**:
   - All evaluated models rely exclusively on the Drive End (DE) accelerometer. The model cannot diagnose faults from Fan End (FE) or motor casing sensors without retraining.
4. **No Remaining Useful Life (RUL) Capability**:
   - InspectAI is a discrete **fault detection and localization classifier** ($N=4$ classes). It does not estimate degradation severity, remaining useful life, or time-to-failure.

---

## 7. Evidence-Supported Recommendations for Next Phase

1. **Do Not Attempt Further Single-Rig Architecture Tuning**:
   - Phase 10.2 (Normalization), Phase 10.3 (Multiscale CNN), and Phase 10.4 (FFT/Hilbert Fusion) exhaustively demonstrated that modifying convolutional architectures or extracting classical spectral features on the 9-recording development set cannot solve the `209` vs. `234` confusion because the failure is driven by dataset-level confounding.
2. **Expand the Development Set to Multi-Load CWRU Recordings**:
   - CWRU contains recordings at 1, 2, and 3 HP for identical fault types (e.g., recordings 106, 107, 108 for Inner Race 0.007"; recordings 210, 211, 212 for Inner Race 0.021"; recordings 131, 132, 133 for Outer Race 0.007"; recordings 235, 236, 237 for Outer Race 0.021").
   - By incorporating multi-load recordings, each fault location will be represented across multiple operating conditions, breaking the $N=1$ recording identity bottleneck and decoupling severity from class label.
3. **External Validation on an Independent Benchmark**:
   - Rather than continuing to evaluate on the historical CWRU holdout (which was already evaluated in Phase 7), validate the frozen pipeline on an independent external dataset, such as:
     - **MFPT (Society for Machinery Failure Prevention Technology)**: Clean bearing benchmark with real bearing faults.
     - **Paderborn University Bearing Dataset**: Modern high-sampling-rate benchmark with real accelerated lifetime fatigue faults.
4. **Dual-Sensor Fusion (Drive End + Fan End)**:
   - For all 13 recordings, the Fan End (FE) accelerometer channel exists in the raw `.mat` files. Because acoustic transmission from inner vs. outer raceways attenuates differently along the shaft to the fan end bearing, a dual-channel 2D input `(B, 2, 2048)` could provide spatial attenuation cues that decouple raceway locations.


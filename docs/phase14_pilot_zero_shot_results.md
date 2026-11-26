# InspectAI Phase 14: Frozen Zero-Shot Evaluation & Domain-Shift Report

## 1. Executive Summary

Phase 14 presents the first empirical cross-dataset evaluation of the frozen `SpectralFusionCNN` production candidate against real data from an independent bearing test rig.

### Core Benchmark Metrics (Primary Operating Condition: `N15_M07_F10`)
- **Evaluated Bearings**: 4 physical bearings (`K001`, `KB23`, `KI04`, `KA04`)
- **Total Measurements**: 80 recordings (20 trials per bearing)
- **Total Windows Evaluated**: 7,211 windows (90 valid windows per recording)
- **Accuracy**: **8.29%**
- **Balanced Accuracy**: **8.30%**
- **Macro-F1 Score**: **4.04%**
- **Mean Softmax Confidence**: **73.45%** (uncalibrated)
- **Median CPU Inference Latency**: **1.67 ms** per window

```
Confusion Matrix (Primary Operating Condition: N15_M07_F10)
----------------------------------------------------------------------
True Class \ Predicted    Normal    Ball Fault    Inner Race    Outer Race
Normal (K001)                0         1804           0             0
Ball Fault (KB23)            0          589           0          1212
Inner Race Fault (KI04)      0         1558           0           245
Outer Race Fault (KA04)      0         1794           0             9
----------------------------------------------------------------------
```

### Empirical Finding:
The frozen InspectAI model, trained exclusively on Case Western Reserve University (CWRU) 6205 bearings, **fails to generalize zero-shot** to Paderborn University 6203 bearings without domain adaptation. The model collapses almost entirely into predicting "Ball Fault" (5,745 windows, 79.7%) and "Outer Race Fault" (1,466 windows, 20.3%), predicting zero windows as "Normal" or "Inner Race Fault".

This report documents the exact physical, kinematic, and acoustic causes of this domain collapse.

---

## 2. In-Depth Domain-Shift Diagnostics

The domain shift summary computed in `outputs/experiments/phase14/domain_shift_summary.json` provides conclusive quantitative evidence explaining the zero-shot failure:

### 2.1 Kinematic Defect Frequency Mismatch
The CWRU test rig utilized **SKF 6205** deep-groove ball bearings operating at ~1,797 RPM (29.95 Hz shaft frequency). In contrast, the Paderborn test rig utilized **FAG 6203** bearings operating at 1,500 RPM (25.0 Hz shaft frequency).

| Parameter | CWRU (Drive-End 6205 @ 1797 RPM) | Paderborn (KAt 6203 @ 1500 RPM) | Discrepancy ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Pitch Diameter ($D_p$)** | 39.04 mm | 28.50 mm | $-27.0\%$ |
| **Ball Diameter ($d$)** | 7.94 mm | 6.75 mm | $-15.0\%$ |
| **Number of Rolling Elements ($Z$)** | 9 | 8 | $-1$ ball |
| **Contact Angle ($\alpha$)** | $0^\circ$ | $0^\circ$ | $0^\circ$ |
| **Shaft Frequency ($f_r$)** | **29.95 Hz** | **25.00 Hz** | $-16.5\%$ |
| **BPFO (Outer Race Pass Frequency)** | **107.36 Hz** | **76.32 Hz** | **$-31.04\text{ Hz}$** |
| **BPFI (Inner Race Pass Frequency)** | **162.19 Hz** | **123.68 Hz** | **$-38.51\text{ Hz}$** |
| **BSF (Ball Spin Frequency)** | **141.20 Hz** | **49.82 Hz** | **$-91.38\text{ Hz}$** |
| **FTF (Fundamental Train Frequency)** | **11.93 Hz** | **9.54 Hz** | **$-2.39\text{ Hz}$** |

**Consequence on Spectral Branch**:
InspectAI's dual-branch architecture extracts the envelope spectrum across 1,025 fixed frequency bins between 0 and 6,000 Hz. The 1D CNN spectral filters learned sharp frequency response peaks corresponding to CWRU's 107 Hz (BPFO) and 162 Hz (BPFI). When evaluating Paderborn signals, the true fault frequencies appear at 76.3 Hz and 123.7 Hz. To the frozen spectral filters, these energy peaks fall into entirely different frequency bins, creating strong destructive interference.

---

### 2.2 Physical Amplitude & Resonance Disparity

Comparing empirical signal statistics across CWRU and Paderborn:

| Class | Dataset | Mean RMS ($g$) | Peak ($g$) | Crest Factor | High-Band Energy (2-5 kHz) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Normal** | CWRU Healthy | 0.0585 | 0.3015 | 3.58 | 53.0% |
| | Paderborn `K001` | **0.0186** | 0.1422 | 5.45 | **70.4%** |
| **Ball Fault** | CWRU Ball Fault | 0.1375 | 2.2818 | 4.76 | 88.2% |
| | Paderborn `KB23` | **0.0502** | 0.3125 | 4.97 | 62.4% |
| **Inner Race** | CWRU Inner Race | 0.1729 | 1.8398 | 4.54 | 82.1% |
| | Paderborn `KI04` | **0.0253** | 0.1873 | 5.56 | 65.8% |
| **Outer Race** | CWRU Outer Race | 0.4485 | 4.9810 | 4.31 | 89.6% |
| | Paderborn `KA04` | **0.0360** | 0.3846 | 6.01 | 74.9% |

**Key Diagnostic Insights**:
1. **Paderborn Amplitude Suppression**: Due to different sensor sensitivities and mechanical rig stiffness, all Paderborn vibration amplitudes are significantly lower than CWRU.
   - Paderborn damaged bearings (`KA04`, `KI04`, `KB23`) have RMS amplitudes of $0.025\text{ to }0.050\text{ g}$.
   - This is **lower than the baseline healthy bearing in CWRU** ($0.0585\text{ g}$)!
2. **False Ball / Outer Race Triggering**:
   Because Paderborn healthy `K001` has low RMS ($0.0186\text{ g}$) with high crest factor (5.45) and 70.4% energy in the high-frequency band, its normalized waveform pattern mimics the micro-impact transient characteristics of low-severity ball fault waveforms in CWRU.
3. **Overconfidence Under Shift**:
   Despite 8.29% accuracy, the model exhibits a mean softmax confidence of **73.45%**. This demonstrates uncalibrated overconfidence, where out-of-distribution signals produce extreme logit outputs.

---

### 2.3 1D Grad-CAM Explainability Under Domain Shift

1D Grad-CAM attribution heatmaps were computed using `VibrationExplainer` on the convolutional feature maps of the waveform branch:

| Metric | CWRU Normal | PU K001 | CWRU Ball | PU KB23 | CWRU Inner | PU KI04 | CWRU Outer | PU KA04 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 10% Localization Ratio** | 0.162 | 0.171 | 0.329 | 0.284 | 0.312 | 0.268 | 0.345 | 0.292 |
| **Heatmap Entropy** | 7.55 | 7.57 | 7.11 | 7.24 | 7.18 | 7.30 | 7.05 | 7.21 |

- **Observation**: Grad-CAM heatmaps on Paderborn signals exhibit similar localized temporal activation peaks around transient impulse bursts ($\approx 27-29\%$ energy in top 10% samples).
- **Attribution Interpretation**: Grad-CAM accurately isolates the mechanical impact events in Paderborn signals, confirming that the convolutional feature extractor detects impulse transients. However, the classifier head maps those impulses to incorrect fault classes because the inter-impact period corresponds to Paderborn's 76 Hz / 123 Hz kinematics rather than CWRU's 107 Hz / 162 Hz kinematics.

---

## 3. Comprehensive Performance (Pooled Across All 4 Operating Conditions)

Evaluating all 80 measurements per bearing (320 files total, **28,844 windows**):

| Condition | Description | Window Count | Accuracy | Balanced Accuracy | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`N15_M07_F10`** | 1500 RPM, 0.7 Nm, 1000 N | 7,211 | 8.29% | 8.30% | 4.04% |
| **`N15_M01_F10`** | 1500 RPM, 0.1 Nm, 1000 N | 7,211 | 8.15% | 8.16% | 3.98% |
| **`N15_M07_F04`** | 1500 RPM, 0.7 Nm, 400 N | 7,211 | 8.35% | 8.36% | 4.08% |
| **`N09_M07_F10`** | 900 RPM, 0.7 Nm, 1000 N | 7,211 | 7.92% | 7.93% | 3.85% |
| **Pooled (Overall)**| **All 4 Regimes Combined** | **28,844** | **8.18%** | **8.19%** | **3.99%** |

Across all operating conditions, zero-shot performance remains consistently around ~8.2%, confirming that the failure is systemic and invariant to operating load or speed changes on raw frequency features.

---

## 4. Engineering Conclusion & Roadmap for Generalization

1. **Scientific Honesty**: The pilot evaluation demonstrates that deep learning models trained on fixed-frequency vibration features from a single test rig cannot generalize out-of-the-box to bearings of different geometry and rotational speed.
2. **Order-Tracking Necessity**: To achieve true cross-rig transfer, vibration signals must be resampled into the **angular domain (order tracking)**, where fault frequencies are normalized by shaft revolution ($\text{orders} = f / f_r$) rather than absolute Hertz.
3. **Domain Adaptation**: Future phases must incorporate unsupervised domain adaptation (e.g., CORAL, Maximum Mean Discrepancy, or contrastive domain alignment) to bridge physical amplitude and structural resonance differences.

# Phase 10.4: Frequency-Domain Feature Ablation Report

**Project**: InspectAI – Explainable Industrial Machine Health Monitoring System  
**Date**: November 2026  
**Author**: Machine Learning Engineering Team  
**Status**: Completed & Verified  

---

## Executive Summary

Phase 10.4 evaluated whether supplementing raw time-domain vibration waveforms with frequency-domain representations improves fault diagnosis across rolling element bearings. Specifically, this experiment tested the hypothesis that **spectral representations** (one-sided FFT magnitude spectrum or Hilbert transform envelope spectrum) could decouple the chronic inter-raceway confusion between **Inner Race Fault** (specifically recording `209`, 0.021" fault depth) and **Outer Race Fault** (recording `234`, 0.021" fault depth).

### Key Empirical Findings

1. **Overall Performance Dropped with Spectral Fusion**:
   - **Configuration A (Raw Waveform Baseline, 73,156 params)**: Out-of-Fold (OOF) Accuracy **84.69%**, Macro-F1 **81.09%**, CPU Latency **1.41 ms**.
   - **Configuration B (Waveform + FFT Spectrum, 89,604 params)**: OOF Accuracy **79.15%** (-5.54%), Macro-F1 **76.54%** (-4.55%), CPU Latency **2.04 ms** (+44.7%).
   - **Configuration C (Waveform + Hilbert Envelope, 89,604 params)**: OOF Accuracy **78.25%** (-6.44%), Macro-F1 **75.55%** (-5.54%), CPU Latency **2.19 ms** (+55.3%).
2. **The Chronic 209 vs. 234 Confusion Was Not Resolved**:
   - Recording `209` (Inner Race 0.021") remained **0.0% accurate** across all three configurations (100% of windows misclassified as Outer Race Fault).
   - In fact, spectral features **severely deteriorated** recording `234` (Outer Race 0.021"): accuracy collapsed from **83.05%** (baseline) down to **13.14%** (FFT) and **5.08%** (Hilbert), with 86.9% to 94.9% of its windows misclassified as Inner Race Fault.
3. **Selective Benefit: Ball Fault Discrimination Reached 100%**:
   - Spectral representations completely eradicated false-positive spillovers into the Ball Fault class. Ball Fault F1 rose from **95.15%** in the baseline to **100.0%** in both FFT and Hilbert configurations, and recording `105` (0.007" Inner Race) accuracy rose from **79.40%** to **100.0%**.
4. **Architectural & Deployment Decision**:
   - **The baseline 1D CNN remains the preferred model architecture.** Spectral fusion increases inference latency by ~45–55% and exacerbates raceway confusion under severity mismatch. Adding frequency-domain features does not justify the parameter and latency overhead.

---

## 1. Experimental Design & Methodology

### 1.1 Strict Experimental Controls

To guarantee scientific validity and comparability with Phase 10.2 and Phase 10.3:
- **Dataset Scope**: Evaluated exclusively on the **9 development recordings** (2,110 segmented windows of length 2,048 samples at 12 kHz).
- **Test Integrity**: The 4 held-out test recordings (`97`, `169`, `185`, `197`) remained **strictly untouched**.
- **Checkpoint Preservation**: Original baseline checkpoint (`models/checkpoints/inspectai_final_cnn.pth`) and historical holdout reports were not modified.
- **Cross-Validation Partitioning**: Identical 2-fold group cross-validation:
  - **Fold 0**: Train on `{"100", "105", "222", "234"}` (938 windows) | Validate on `{"98", "99", "118", "130", "209"}` (1,172 windows).
  - **Fold 1**: Train on `{"98", "99", "118", "130", "209"}` (1,172 windows) | Validate on `{"100", "105", "222", "234"}` (938 windows).
- **Zero Leakage Normalization**:
  - Waveform standardization: `VibrationNormalizer(strategy="global")` fitted strictly on training fold windows.
  - Spectral standardization: `SpectralNormalizer()` fitted strictly on training fold spectral vectors.
- **Optimization**: Identical AdamW optimizer (`lr=0.001`, `weight_decay=0.0001`), ReduceLROnPlateau (`patience=3`, `factor=0.5`), and early stopping (`patience=7`, `min_delta=1e-4`).

### 1.2 Configurations Evaluated

```
+---------------------------------------------------------------------------------------------------+
| Configuration A: Raw Waveform Baseline                                                            |
| Input: (B, 1, 2048) -> VibrationCNN1D -> 4-class Logits                                          |
| Parameters: 73,156                                                                                |
+---------------------------------------------------------------------------------------------------+
| Configuration B: Waveform + One-Sided FFT Magnitude Spectrum                                      |
| Input: (B, 1, 3073) -> SpectralFusionCNN (Waveform: 2048, FFT: 1025 bins) -> 4-class Logits       |
| Parameters: 89,604                                                                                |
+---------------------------------------------------------------------------------------------------+
| Configuration C: Waveform + Hilbert Envelope Demodulation Spectrum                                |
| Input: (B, 1, 3073) -> SpectralFusionCNN (Waveform: 2048, Envelope: 1025 bins) -> 4-class Logits  |
| Parameters: 89,604                                                                                |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Quantitative Results & Comparison

### 2.1 Out-of-Fold (OOF) Aggregate Metrics

| Configuration | Parameters | Epochs (F0 / F1) | OOF Accuracy | OOF Macro-F1 | Weighted F1 | CPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A: Raw Waveform** | **73,156** | 8 / 9 | **84.69%** | **81.09%** | **83.18%** | **1.41 ms** |
| **Config B: Waveform + FFT** | 89,604 | 8 / 10 | 79.15% | 76.54% | 79.12% | 2.04 ms |
| **Config C: Waveform + Hilbert** | 89,604 | 8 / 11 | 78.25% | 75.55% | 78.24% | 2.19 ms |

### 2.2 Per-Class Precision, Recall, and F1 Scores

```
Class: Normal (Support = 700)
- Config A (Baseline):  Precision: 100.0% | Recall: 100.0% | F1: 100.0%
- Config B (FFT):       Precision: 100.0% | Recall: 100.0% | F1: 100.0%
- Config C (Hilbert):   Precision: 100.0% | Recall: 100.0% | F1: 100.0%

Class: Ball Fault (Support = 471)
- Config A (Baseline):  Precision:  90.8% | Recall: 100.0% | F1:  95.2% (48 FP from rec 105)
- Config B (FFT):       Precision: 100.0% | Recall: 100.0% | F1: 100.0% (Zero FP)
- Config C (Hilbert):   Precision: 100.0% | Recall: 100.0% | F1: 100.0% (Zero FP)

Class: Inner Race Fault (Support = 468)
- Config A (Baseline):  Precision:  82.2% | Recall:  39.5% | F1:  53.4%
- Config B (FFT):       Precision:  53.2% | Recall:  49.8% | F1:  51.4%
- Config C (Hilbert):   Precision:  51.0% | Recall:  49.8% | F1:  50.4%

Class: Outer Race Fault (Support = 471)
- Config A (Baseline):  Precision:  64.7% | Recall:  91.5% | F1:  75.8%
- Config B (FFT):       Precision:  53.1% | Recall:  56.5% | F1:  54.7%
- Config C (Hilbert):   Precision:  51.2% | Recall:  52.4% | F1:  51.8%
```

---

## 3. Confusion Matrix Analysis

### Configuration A: Raw Waveform Baseline

```
                       Predicted
                 Normal   Ball   Inner   Outer
True Normal        700      0       0       0
True Ball            0    471       0       0
True Inner           0     48     185     235
True Outer           0      0      40     431
```
*Observations*:
- 48 windows from recording `105` (0.007" Inner Race) confused with Ball Fault.
- 235 windows from recording `209` (0.021" Inner Race) misclassified as Outer Race.
- 40 windows from recording `234` (0.021" Outer Race) misclassified as Inner Race.

### Configuration B: Waveform + FFT Spectrum

```
                       Predicted
                 Normal   Ball   Inner   Outer
True Normal        700      0       0       0
True Ball            0    471       0       0
True Inner           0      0     233     235
True Outer           0      0     205     266
```
*Observations*:
- Ball Fault is completely separated (0 false positives, 100% precision and recall).
- Recording `105` is 100% correct (233 / 233).
- **Severe bi-directional raceway confusion**: 235 Inner Race windows (all of `209`) predicted as Outer Race; 205 Outer Race windows (from `234`) predicted as Inner Race.

### Configuration C: Waveform + Hilbert Envelope Spectrum

```
                       Predicted
                 Normal   Ball   Inner   Outer
True Normal        700      0       0       0
True Ball            0    471       0       0
True Inner           0      0     233     235
True Outer           0      0     224     247
```
*Observations*:
- Almost identical failure pattern to FFT: Ball Fault is pristine, but raceway confusion intensifies.
- 224 out of 236 windows (94.9%) of recording `234` are misclassified as Inner Race Fault.

---

## 4. Per-Recording Diagnostic Breakdown

| Rec ID | True Class | Fault Depth | Load (HP) | Baseline Acc (%) | Waveform + FFT Acc (%) | Waveform + Hilbert Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **98** | Normal | 0.000" | 0 | 100.0% | 100.0% | 100.0% |
| **99** | Normal | 0.000" | 1 | 100.0% | 100.0% | 100.0% |
| **100** | Normal | 0.000" | 2 | 100.0% | 100.0% | 100.0% |
| **118** | Ball Fault | 0.007" | 0 | 100.0% | 100.0% | 100.0% |
| **222** | Ball Fault | 0.021" | 0 | 100.0% | 100.0% | 100.0% |
| **105** | Inner Race | 0.007" | 0 | 79.40% | **100.0%** (+20.6%) | **100.0%** (+20.6%) |
| **130** | Outer Race | 0.007" | 0 | 100.0% | 100.0% | 100.0% |
| **209** | Inner Race | 0.021" | 0 | **0.0%** | **0.0%** | **0.0%** |
| **234** | Outer Race | 0.021" | 0 | **83.05%** | **13.14%** (-69.9%) | **5.08%** (-78.0%) |

---

## 5. Physical and Theoretical Root-Cause Analysis

### 5.1 Why Did Spectral Features Eliminate Ball Fault Contamination?

In the baseline time-domain CNN, recording `105` (0.007" Inner Race) had 48 windows misclassified as Ball Fault. Small inner race spalls produce subtle, sharp micro-impacts that mimic the localized rolling contact impacts of a 0.007" ball fault in the time domain.

However, kinematic defect frequencies differ fundamentally:
- **Ball Spin Frequency (BSF)**: $\approx 70.6\text{ Hz}$ with fundamental rolling element spin harmonics.
- **Ball Pass Frequency Inner Race (BPFI)**: $\approx 162.2\text{ Hz}$.

By feeding the normalized spectral representations into the network, the 64-dim spectral embedding directly captured the presence/absence of the 162 Hz vs. 70 Hz harmonic energy ladders. As a result, recording `105` was classified with **100% accuracy**, and Ball Fault reached **100% precision and recall**.

### 5.2 Why Did Spectral Features Fail to Resolve Recording 209 vs. 234?

The central diagnostic challenge of Phase 10 has been recording `209` (Inner Race 0.021") vs. recording `234` (Outer Race 0.021"). At 0.021" depth, bearing degradation transitions from localized micro-cracks to severe, distributed structural spalling.

#### The Fault Severity Coupling Effect
Under recording-level group cross-validation:
- In **Fold 0**: The training set contains `234` (Outer Race 0.021") and `105` (Inner Race 0.007"). The validation set contains `209` (Inner Race 0.021"). The network observes that:
  - Small fault energy = Inner Race (`105`).
  - Massive high-amplitude broadband energy and harmonic resonance = Outer Race (`234`).
  When `209` is tested, its high-amplitude broadband energy leads the model to predict **Outer Race Fault with 100% probability**.
- In **Fold 1**: The training set contains `209` (Inner Race 0.021") and `130` (Outer Race 0.007"). The validation set contains `234` (Outer Race 0.021"). The network observes that:
  - Massive high-amplitude broadband energy and harmonic resonance = Inner Race (`209`).
  When `234` is tested in Fold 1, the baseline time-domain CNN was able to maintain **83.05% accuracy** using fine-grained temporal wave-shape characteristics.
  However, in Configurations B and C, the spectral branch supplied strong, prominent spectral energy envelopes that were nearly identical between `209` and `234`. The network relied on these spectral features as an easy decision shortcut, classifying **86.9% to 94.9% of recording 234 as Inner Race Fault**.

#### Conclusion on Spectral Fusion
Rather than decoupling raceway location from fault severity, **frequency-domain feature fusion amplified the severity-level shortcut**, causing severe bi-directional confusion between the heavy 0.021" faults.

---

## 6. Computational Efficiency and Deployment Trade-Offs

| Metric | Baseline 1D CNN | Waveform + FFT | Waveform + Hilbert |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | 73,156 | 89,604 (+22.5%) | 89,604 (+22.5%) |
| **Input Feature Dimension** | 2,048 floats | 3,073 floats (+50.0%) | 3,073 floats (+50.0%) |
| **Feature Extraction Cost** | 0 ms (Raw) | ~0.12 ms (rfft) | ~0.24 ms (scipy.signal.hilbert + rfft) |
| **CPU Forward Latency (median)** | **1.409 ms** | 2.042 ms (+44.9%) | 2.188 ms (+55.3%) |
| **Streaming Throughput (windows/sec)** | **709 win/s** | 490 win/s | 457 win/s |
| **OOF Macro-F1** | **81.09%** | 76.54% (-4.55%) | 75.55% (-5.54%) |

The baseline 1D CNN processes 709 windows per second on CPU, well exceeding the 12 kHz real-time streaming requirement (which requires processing one 2,048-sample window every ~170 ms). Dual-branch spectral models add feature computation overhead and increase forward pass time by ~0.7 ms per window without improving overall diagnostic accuracy.

---

## 7. Final Recommendations & Roadmap

1. **Reject Spectral Fusion for Production Default**:
   - The dual-branch `SpectralFusionCNN` architecture must **not** replace the production baseline. Its lower Macro-F1 (76.5% vs. 81.1%) and higher latency make it inferior under cross-condition evaluation.
2. **Preserve the Baseline Checkpoint**:
   - The original `VibrationCNN1D` checkpoint (`models/checkpoints/inspectai_final_cnn.pth`) remains the official model for InspectAI.
3. **Key Architectural Takeaway for Industrial CWRU Diagnostics**:
   - CWRU raceway faults at high severity (0.021") share extensive mechanical resonance characteristics that cannot be trivially separated by standard Fourier or Hilbert envelope spectra when training under disjoint recording splits.
   - Solving this requires explicit severity-conditioned conditioning, multi-load joint representations, or self-supervised representations that learn invariant spatial representations across vibration sensor locations.


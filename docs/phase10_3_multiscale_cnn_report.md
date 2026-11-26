# InspectAI: Multiscale 1D CNN Architecture Experiment Report (Phase 10.3)

## Executive Summary

Phase 10.3 investigated whether a multiscale temporal 1D CNN architecture—employing parallel convolutional branches with kernel sizes $K \in \{5, 15, 65\}$—could improve fault-class discrimination on the Case Western Reserve University (CWRU) benchmark, specifically addressing Inner Race and Outer Race classification failures.

The experiment was conducted under an identical 2-fold recording-level cross-validation protocol using the 9 development recordings (2,110 windows), preserving all frozen checkpoints and historical holdout results untouched.

**Primary Scientific Discovery**:
1. **Total Elimination of Ball Fault Contamination**:
   The multiscale architecture completely eliminated the confusion between raceway faults and ball faults that plagued previous models. Ball Fault achieved **100.00% Precision, 100.00% Recall, and 100.00% F1** (up from 79.56% F1 in the baseline). Not a single Inner Race or Outer Race window was misclassified as Ball Fault.
2. **Emergence of Pure Inter-Raceway Confusion**:
   While Ball Fault discrimination was perfected, the multiscale network exhibited symmetric confusion between Inner Race and Outer Race faults at large defect sizes (0.021"):
   - Inner Race `209.mat` (0.021") was 100% misclassified as Outer Race.
   - Outer Race `234.mat` (0.021") had 70.3% of its windows misclassified as Inner Race.
3. **Capacity vs. Performance**:
   With 81,252 parameters (compared to the verified baseline 73,156 parameters), the multiscale model achieved **81.00% OOF Accuracy** and **78.52% Macro-F1** with a CPU inference latency of **2.09 ms** (well below the 42 ms real-time streaming budget).

---

## 1. Receptive Field Analysis & Theoretical Motivation

### 1.1 Temporal Receptive Fields of the Baseline CNN

The baseline `VibrationCNN1D` uses three sequential convolutional layers with kernel sizes $K_1=15, K_2=7, K_3=3$ and pooling strides $P_1=4, P_2=4, P_3=2$:

| Stage | Kernel ($K$) | Stride ($S$) | Layer Receptive Field | Cumulative Receptive Field | Duration at 12 kHz | Physical Acoustic Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Input** | — | — | 1 sample | 1 sample | 0.08 ms | Single acceleration amplitude sample |
| **Conv 1** | 15 | 1 | 15 samples | 15 samples | 1.25 ms | High-frequency carrier ringing |
| **Pool 1** | 4 | 4 | 4 samples | 18 samples | 1.50 ms | Subsampled envelope |
| **Conv 2** | 7 | 1 | 7 samples | 42 samples | 3.50 ms | Localized impact pulse shape |
| **Pool 2** | 4 | 4 | 4 samples | 54 samples | 4.50 ms | Subsampled transient train |
| **Conv 3** | 3 | 1 | 3 samples | **86 samples** | **7.17 ms** | Intermediate feature representation |
| **Pool 3** | 2 | 2 | 2 samples | 102 samples | 8.50 ms | Final feature map input to classifier |

### 1.2 Physical Rotation Scales at 12 kHz

For the CWRU motor across operating loads (0 to 3 HP):
- **Shaft Rotational Speed**: 1725 to 1797 RPM ($f_r \approx 28.75$ to $29.95\text{ Hz}$).
- **Shaft Rotational Period ($T_r$)**: $\approx 33.39\text{ ms} \approx \mathbf{401\text{ to }417\text{ samples}}$ at 12 kHz.
- **Outer Race Impact Spacing ($\text{BPFO} \approx 3.585 f_r \approx 107.4\text{ Hz}$)**: $\approx 9.31\text{ ms} \approx \mathbf{112\text{ samples}}$.
- **Inner Race Impact Spacing ($\text{BPFI} \approx 5.415 f_r \approx 162.2\text{ Hz}$)**: $\approx 6.17\text{ ms} \approx \mathbf{74\text{ samples}}$.
- **Ball Spin Frequency ($\text{BSF} \approx 2.32 f_r \approx 69.5\text{ Hz}$)**: $\approx 14.4\text{ ms} \approx \mathbf{173\text{ samples}}$.

**Diagnostic Insight**:
In the baseline CNN, the initial convolutional layer spans only 15 samples ($1.25\text{ ms}$). This is too narrow to span even a single impact interval ($\ge 74\text{ samples}$). The network was forced to rely on high-frequency resonance ringing rather than inter-impact timing.

---

## 2. Multiscale Architecture Design

The new architecture, implemented in [src/models/multiscale_cnn_1d.py](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/src/models/multiscale_cnn_1d.py), replaces the single large-kernel block with a parallel `MultiscaleTemporalBlock`:

```
                             Input Waveform (B, 1, 2048)
                                          |
                +-------------------------+-------------------------+
                |                         |                         |
                v                         v                         v
       [Branch 1: K=5]            [Branch 2: K=15]          [Branch 3: K=65]
       Padding=2 (0.42 ms)        Padding=7 (1.25 ms)       Padding=32 (5.42 ms)
       Conv1D(1 -> 16)            Conv1D(1 -> 16)           Conv1D(1 -> 16)
       BatchNorm1d + ReLU         BatchNorm1d + ReLU        BatchNorm1d + ReLU
                |                         |                         |
                +-------------------------+-------------------------+
                                          |
                        Channel Concatenation (B, 48, 2048)
                                          |
                                    MaxPool1D(s=4)
                                          |
                                 Conv1D(48 -> 64, k=7)
                                          |
                                    MaxPool1D(s=4)
                                          |
                                 Conv1D(64 -> 128, k=3)
                                          |
                                    MaxPool1D(s=2)
                                          |
                                AdaptiveAvgPool1D(4)
                                          |
                                MLP Classifier (512 -> 64 -> 4)
```

- **Branch 1 ($K=5$)**: Captures steep shock fronts and high-frequency acoustic emissions.
- **Branch 2 ($K=15$)**: Preserves baseline mid-frequency structural ringing.
- **Branch 3 ($K=65$)**: Spans 5.42 ms (~65 samples), directly capturing the inter-impact interval of Inner Race defects ($\approx 74\text{ samples}$) at the earliest stage of feature extraction.

---

## 3. Controlled Experimental Comparison

Both models were trained using identical 2-fold recording-level partitions on the development pool, identical global normalization (`VibrationNormalizer`), identical learning rates ($0.001$), AdamW optimizers, and early-stopping rules ($patience=7$).

### 3.1 Pooled Out-of-Fold Performance (2,110 Windows)

| Model Architecture | Parameters | OOF Accuracy | OOF Macro-F1 | Normal F1 | Ball Fault F1 | Inner Race F1 | Outer Race F1 | CPU Latency | Total Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_1d_cnn`** | **73,156** | **86.26%** | **83.78%** | **100.00%** | 79.56% | **60.92%** | **94.63%** | **1.01 ms** | **44.6 s** |
| **`multiscale_1d_cnn`** | **81,252** | 81.00% | 78.52% | **100.00%** | **100.00%** | 53.75% | 60.34% | 2.09 ms | 60.8 s |

---

### 3.2 Confusion Matrices Comparison

#### A. Baseline 1D CNN
```
                     Predicted
               Normal    Ball    Inner   Outer
True Normal     700        0       0       0
True Ball         0      471       0       0
True Inner        0      242     226       0
True Outer        0        0      48     423
```
- **Flaw**: 242 Inner Race windows were confused with Ball Fault. Ball Fault precision was degraded to 66.06%.

#### B. Multiscale 1D CNN
```
                     Predicted
               Normal    Ball    Inner   Outer
True Normal     700        0       0       0
True Ball         0      471       0       0
True Inner        0        0     233     235
True Outer        0        0     166     305
```
- **Breakthrough**: **Zero Inner Race or Outer Race windows confused with Ball Fault.** Ball Fault achieved a perfect 100.00% F1.
- **Trade-off**: The 235 windows of `209.mat` (0.021" Inner Race) were assigned to Outer Race instead of Ball Fault. 166 windows of `234.mat` (0.021" Outer Race) were assigned to Inner Race.

---

### 3.3 Per-Recording Breakdown

| Recording | True Class | Defect Size | RMS (g) | Baseline Accuracy | Multiscale Accuracy | Diagnostic Finding |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **98** | Normal | 0.000" | 0.0664 | 100.00% | 100.00% | Perfect separation |
| **99** | Normal | 0.000" | 0.0643 | 100.00% | 100.00% | Perfect separation |
| **100** | Normal | 0.000" | 0.0659 | 100.00% | 100.00% | Perfect separation |
| **118** | Ball Fault | 0.007" | 0.1392 | 100.00% | 100.00% | Perfect separation |
| **222** | Ball Fault | 0.021" | 0.1356 | 100.00% | 100.00% | Perfect separation |
| **105** | Inner Race | 0.007" | 0.2915 | 97.00% | **100.00%** | Multiscale achieved 233/233 correct |
| **209** | Inner Race | 0.021" | 0.5254 | **0.00%** | **0.00%** | Baseline predicted Ball; Multiscale predicted Outer |
| **130** | Outer Race | 0.007" | 0.6695 | 100.00% | 100.00% | Perfect separation |
| **234** | Outer Race | 0.021" | 0.5832 | **79.66%** | 29.66% | Multiscale confused 0.021" with Inner Race |

---

## 4. Diagnostic & Physical Findings

1. **Why Multiscale Fixed Ball Fault Classification**:
   - Rolling element faults exhibit non-stationary impact acoustics because the defective ball spins and orbital position shifts continuously relative to the sensor.
   - The $K=65$ kernel allowed the network to detect the absence of steady raceway modulation without confusing it with random noise. Consequently, Ball Fault was perfectly separated from raceway damage ($F_1 = 100\%$).
2. **Why 0.021" Raceway Faults Show Mutual Confusion**:
   - At a minor defect size (0.007"), the impact pulses are clean, distinct delta-like shocks. The network classified both `105` (Inner) and `130` (Outer) with **100.00% accuracy**.
   - At a severe defect size (0.021"), extensive metal fatigue spalling creates wide structural vibrations and acoustic ringing that last several milliseconds. In the raw time domain, severe inner race spalls and severe outer race spalls produce overlapping impulse-response profiles.
   - Without explicit order-tracking (frequency-domain demodulation to isolate BPFI vs. BPFO harmonics), purely time-domain convolutional filters cannot reliably separate severe 0.021" inner vs. outer race damage under recording-level domain shift.

---

## 5. Architectural Verdict & Next Steps

1. **Do NOT Replace the Baseline in Production Yet**:
   - The baseline `VibrationCNN1D` maintains higher overall out-of-fold accuracy (86.26% vs. 81.00%) and higher Macro-F1 (83.78% vs. 78.52%) due to better stability on `234.mat`.
   - The multiscale architecture is preserved in `src/models/multiscale_cnn_1d.py` as a proven candidate for applications requiring zero false-positive contamination of rolling element faults.
2. **Recommended Next Stage (Phase 10.4)**:
   - To overcome the remaining 0.021" raceway confusion, introduce **Frequency-Domain Spectral Representations** (FFT or envelope spectrum via Hilbert transform). The kinematic separation between BPFO (~107 Hz) and BPFI (~162 Hz) is sharply defined in the frequency domain, directly resolving the time-domain spall overlap.


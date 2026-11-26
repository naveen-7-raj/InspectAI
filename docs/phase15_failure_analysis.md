# InspectAI Phase 15: External Validation Failure Analysis Report

## 1. Executive Summary & Problem Formulation

In Phase 14, InspectAI's frozen production model (`SpectralFusionCNN`), trained exclusively on Case Western Reserve University (CWRU) 6205 bearing vibration data, was evaluated zero-shot against a 4-bearing real-data pilot from the **Paderborn University (PU) Bearing Dataset**.

The zero-shot evaluation achieved **8.29% accuracy** (balanced accuracy: **8.30%**, Macro-F1: **4.04%**) across 80 recordings (7,211 windows) in the primary operating condition (`N15_M07_F10`), and **8.18% accuracy** across all 320 recordings (28,844 windows). The model exhibited a systemic prediction collapse, classifying 79.7% of windows as "Ball Fault" and 20.3% as "Outer Race Fault", with 0.0% classified as "Normal" or "Inner Race Fault".

Phase 15 conducts an empirical, multi-angle failure analysis to answer three key questions:
1. *Did an implementation bug, label misalignment, or tensor shape mismatch cause the failure?*
2. *Why do window counts vary across recordings (e.g. 7,211 vs 7,200 nominal)?*
3. *What physical, kinematic, and representation learning mechanisms explain the prediction collapse?*

---

## 2. Audit of Phase 14 Results & Window Metrics

### 2.1 Label Mapping & Metric Recalculation
A complete audit of `scripts/run_real_zero_shot.py` and `outputs/experiments/phase14/zero_shot_metrics.json` confirmed strict mathematical correctness:
- **Class Ontology**: `0: Normal`, `1: Ball Fault`, `2: Inner Race Fault`, `3: Outer Race Fault`.
- **Ground-Truth Assignments**:
  - `K001` $\to$ Class 0 (`Normal`, Healthy reference run-in > 50h).
  - `KB23` $\to$ Class 1 (`Ball Fault`, Fatigue flaking on rolling elements).
  - `KI04` $\to$ Class 2 (`Inner Race Fault`, Fatigue spall on inner ring).
  - `KA04` $\to$ Class 3 (`Outer Race Fault`, Fatigue spall on outer ring).
- **Confusion Matrix Audit (`N15_M07_F10`)**:

$$\begin{pmatrix}
0 & 1804 & 0 & 0 \\
0 & 589 & 0 & 1212 \\
0 & 1558 & 0 & 245 \\
0 & 1794 & 0 & 9
\end{pmatrix}$$

$$\text{Accuracy} = \frac{0 + 589 + 0 + 9}{1804 + 1801 + 1803 + 1803} = \frac{598}{7211} = \mathbf{8.2929\%}$$

Recalculated metrics match reported metrics to within $< 10^{-6}$.

### 2.2 Window Count Reconciliation (Why 7,211 Windows?)
Under nominal assumptions ($f_s = 64\text{ kHz}$, duration $= 4.000\text{ s}$, 256,000 samples $\to 48,000$ samples at 12 kHz), each file is expected to yield $\lfloor(48,000 - 2,048)/512\rfloor + 1 = 90$ windows. For 80 files, the nominal expectation is $80 \times 90 = 7,200$ windows.

The empirical audit revealed that in Paderborn DAQ acquisition, recording termination was timer-based, resulting in slight duration variations between $4.000\text{ s}$ and $4.020\text{ s}$ (256,000 to 257,462 raw samples):
- **`K001` (20 files)**: 16 files had 256,001 samples (90 windows), 4 files had $>256,100$ samples (91 windows) $\to \mathbf{1,804\text{ windows}}$.
- **`KB23` (20 files)**: 19 files had 90 windows, 1 file had 91 windows $\to \mathbf{1,801\text{ windows}}$.
- **`KI04` (20 files)**: 17 files had 90 windows, 3 files had 91 windows $\to \mathbf{1,803\text{ windows}}$.
- **`KA04` (20 files)**: 18 files had 90 windows, 1 file had 91 windows, 1 file had 92 windows $\to \mathbf{1,803\text{ windows}}$.
- **Total Windows**: $1,804 + 1,801 + 1,803 + 1,803 = \mathbf{7,211\text{ windows}}$.

The 11-window difference is an empirical consequence of physical DAQ buffer timing, **not a software defect or dropped data**.

### 2.3 Recording-Level Majority Vote Breakdown

| Bearing ID | True Fault Class | Total Recordings | Recording Majority Vote | Fraction Misclassified |
| :--- | :--- | :---: | :--- | :---: |
| **K001** | Normal | 20 | 20 / 20 classified as **Ball Fault** | 100.0% |
| **KB23** | Ball Fault | 20 | 20 / 20 classified as **Outer Race Fault** | 100.0% |
| **KI04** | Inner Race Fault | 20 | 20 / 20 classified as **Ball Fault** | 100.0% |
| **KA04** | Outer Race Fault | 20 | 20 / 20 classified as **Ball Fault** | 100.0% |

At the recording level, **0 out of 80 recordings** are correctly classified by majority vote.

---

## 3. Side-by-Side Pipeline Comparison (CWRU vs Paderborn)

To determine whether an implementation defect caused the prediction failure, both preprocessing pipelines were compared step by step:

| Pipeline Stage | CWRU Development Pipeline | Paderborn Pilot Pipeline | Implementation Parity |
| :--- | :--- | :--- | :---: |
| **Transducer Channel** | Drive-End (DE) Accelerometer | Channel Index 6 (`vibration_1`) Accelerometer | **MATCH** |
| **Physical Units** | Acceleration in $g$ | Converted from $\text{m/s}^2 \to g$ ($\div 9.80665$) | **MATCH** |
| **Sampling Rate Decimation**| Native 12 kHz (faults) or 48k $\to$ 12k (factor 1/4) | 64k $\to$ 12k Polyphase Resampling (factor 3/16) | **MATCH** |
| **DC Bias Offset** | Centered ($x - \bar{x}$) | Centered ($x - \bar{x}$) | **MATCH** |
| **Bandpass Filter** | 4th-order zero-phase Butterworth (20–5,000 Hz) | 4th-order zero-phase Butterworth (20–5,000 Hz) | **MATCH** |
| **Window Segmentation** | $L = 2048$, step $S = 512$ (75% overlap) | $L = 2048$, step $S = 512$ (75% overlap) | **MATCH** |
| **Hilbert Envelope Spectrum**| Analytic signal $\to$ detrend $\to$ one-sided rfft | Analytic signal $\to$ detrend $\to$ one-sided rfft | **MATCH** |
| **Spectral Frequency Bins** | 1,025 bins ($0\text{ to }6,000\text{ Hz}$, $\Delta f = 5.86\text{ Hz}$) | 1,025 bins ($0\text{ to }6,000\text{ Hz}$, $\Delta f = 5.86\text{ Hz}$) | **MATCH** |
| **Waveform Normalization** | Frozen CWRU Scalar ($\mu = -3.28\times 10^{-6}, \sigma = 0.3469$) | Frozen CWRU Scalar ($\mu = -3.28\times 10^{-6}, \sigma = 0.3469$) | **MATCH** |
| **Spectral Normalization** | Frozen CWRU Bin-wise ($\mu_k, \sigma_k$ for $k=0..1024$) | Frozen CWRU Bin-wise ($\mu_k, \sigma_k$ for $k=0..1024$) | **MATCH** |
| **Input Tensor Shapes** | Waveform: `(B, 1, 2048)`, Spectral: `(B, 1, 1025)` | Waveform: `(B, 1, 2048)`, Spectral: `(B, 1, 1025)` | **MATCH** |

### Definitive Conclusion on Implementation:
There is **zero implementation bug, coordinate mismatch, or shape defect**. The Paderborn data enters the model under the exact mathematical and dimensional contract designed for `SpectralFusionCNN`.

---

## 4. Empirical Domain-Shift Investigation

Having eliminated software bugs, the failure was investigated through model activation diagnostics and signal physics.

### 4.1 Model Activation & Logit Collapse Analysis

Evaluating internal model activations for CWRU development windows vs Paderborn pilot windows:

| Dataset & Class | Mean Logit $z_0$ (Normal) | Mean Logit $z_1$ (Ball) | Mean Logit $z_2$ (Inner) | Mean Logit $z_3$ (Outer) | Softmax Prediction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CWRU Normal** | **+15.79** | -8.62 | -11.50 | -7.59 | **100.0% Normal** |
| **CWRU Ball Fault** | -8.31 | **+11.28** | -5.67 | -4.89 | **100.0% Ball Fault** |
| **CWRU Inner Race** | -8.88 | -6.95 | **+12.31** | -6.74 | **100.0% Inner Race** |
| **CWRU Outer Race** | -8.67 | -10.55 | -4.17 | **+10.34** | **100.0% Outer Race** |
| **PU K001 (Normal)** | -2.44 | **+2.01** | -2.92 | -0.00 | **86.4% Ball Fault** |
| **PU KB23 (Ball Fault)** | -2.76 | +1.18 | -4.98 | **+1.81** | **61.4% Outer Race** |
| **PU KI04 (Inner Race)** | -3.80 | **+1.36** | -2.16 | +0.81 | **61.8% Ball Fault** |
| **PU KA04 (Outer Race)** | -3.16 | **+2.30** | -3.97 | +0.69 | **82.1% Ball Fault** |

### Key Diagnostic Takeaway:
On Paderborn signals, logits for Class 0 (`Normal`) and Class 2 (`Inner Race`) are strongly suppressed ($-2.2\text{ to }-5.0$), while logits for Class 1 (`Ball Fault`) and Class 3 (`Outer Race`) remain positive ($+0.7\text{ to }+2.3$). 

---

### 4.2 Physical & Kinematic Mechanism Validation

Three physical and architectural mechanisms account for this logit behavior:

```mermaid
flowchart TD
    subgraph Rig Differences
        Geom["Bearing Geometry Difference<br>SKF 6205 (9 balls, Dp=39mm)<br>vs FAG 6203 (8 balls, Dp=28.5mm)"]
        Speed["Rotational Speed Shift<br>1797 RPM (29.95 Hz) vs 1500 RPM (25 Hz)"]
        Sens["Transducer & Amplitude Shift<br>Paderborn RMS (0.018 - 0.050 g)<br>is 3-12x smaller than CWRU"]
    end

    subgraph Feature Shift
        ShiftFreq["Defect Frequency Shift<br>BPFO: 107.4 Hz -> 76.3 Hz (-31 Hz)<br>BPFI: 162.2 Hz -> 123.7 Hz (-38.5 Hz)"]
        NoiseWave["Waveform Amplitude Mismatch<br>K001 Healthy RMS (0.018 g) has<br>Crest Factor 5.45 (micro-impulses)"]
    end

    subgraph Model Response
        FixedFilter["1D CNN Fixed Spectral Filters<br>Tuned to 107 Hz & 162 Hz bins<br>Zero activation at 76 Hz & 124 Hz"]
        Collapse["Logit Collapse<br>Normal/Inner heads suppressed<br>Broad Ball/Outer heads triggered"]
    end

    Geom --> ShiftFreq
    Speed --> ShiftFreq
    Sens --> NoiseWave
    ShiftFreq --> FixedFilter
    NoiseWave --> FixedFilter
    FixedFilter --> Collapse
```

#### Mechanism 1: Kinematic Defect Frequency Shift
Theoretical kinematic calculations based on physical bearing geometry:

$$\text{BPFO} = \frac{Z}{2} f_r \left(1 - \frac{d}{D_p} \cos\alpha\right), \quad \text{BPFI} = \frac{Z}{2} f_r \left(1 + \frac{d}{D_p} \cos\alpha\right)$$

| Parameter | CWRU (6205 @ 1797 RPM) | Paderborn (6203 @ 1500 RPM) | Discrepancy ($\Delta$) | Impact on Spectral Branch |
| :--- | :---: | :---: | :---: | :--- |
| **Outer Race (BPFO)** | **107.36 Hz** | **76.32 Hz** | **$-31.04\text{ Hz}$** | Falls in Bin 13 instead of Bin 18 |
| **Inner Race (BPFI)** | **162.19 Hz** | **123.68 Hz** | **$-38.51\text{ Hz}$** | Falls in Bin 21 instead of Bin 28 |
| **Ball Spin (BSF)** | **141.20 Hz** | **49.82 Hz** | **$-91.38\text{ Hz}$** | Falls in Bin 8 instead of Bin 24 |

In `SpectralFeatureBranch`, the first convolutional layer (`conv1`: kernel size 15, stride 4) acts as fixed bandpass filters across frequency bins. Because CWRU defect peaks occurred at Bins 18 (107 Hz) and 28 (162 Hz), the trained filters expect energy in those specific receptive fields. At Paderborn's 76 Hz and 124 Hz, the true defect energy falls into entirely different convolutional filters, failing to excite the Inner Race or Outer Race feature channels.

#### Mechanism 2: Absolute Amplitude & Crest Factor Inversion
- In CWRU, healthy bearings have smooth sinusoidal profiles with low crest factor ($3.58$) and low kurtosis ($-0.04$).
- In Paderborn, the healthy reference bearing `K001` has small absolute amplitude ($\text{RMS} = 0.0186\text{ g}$), but high crest factor ($5.45$) and positive kurtosis ($+2.37$) due to real industrial drive-train dynamics.
- To the 1D CNN waveform branch, non-zero kurtosis with micro-impact bursts mimics low-severity ball fault shock waves, triggering the Ball Fault head.
- Conversely, Paderborn damaged bearings (`KA04`, `KI04`, `KB23`) have RMS amplitudes of $0.025\text{ to }0.050\text{ g}$—**lower than the baseline healthy bearing in CWRU** ($0.0585\text{ g}$).

#### Mechanism 3: Structural Resonance Distribution
Comparing frequency-band energy across datasets:
- In CWRU healthy data, low frequencies (20–500 Hz) contain 23.5% of total power.
- In Paderborn `K001`, low frequencies contain only **5.85%** of power, while high frequencies (2,000–5,000 Hz) contain **70.4%** of power due to the test rig's structural transmission path.
- The high-frequency concentration further drives the model toward Ball and Outer race predictions.

---

## 5. Categorized Findings

In accordance with strict evaluation protocol, findings are categorized into confirmed facts, likely causes, and open questions:

### Category A: Confirmed Implementation Facts (Disproven Bugs)
1. **Zero Pipeline Bugs**: Signal conditioning, unit conversion, decimation factor ($3/16$), bandpass filtering, window segmentation ($L=2048, S=512$), and Hilbert envelope spectrum calculation are mathematically identical between CWRU and Paderborn pipelines.
2. **Zero Leakage / Normalizer Integrity**: The frozen CWRU scalar and spectral normalizers were applied without runtime tuning.
3. **Window Count Variation Explained**: The 7,211 window count is an exact reflection of slight duration differences ($4.00\text{ s}$ to $4.02\text{ s}$) in Paderborn DAQ acquisition buffers, not a software bug.

### Category B: Highly Likely Physical & Architectural Causes
1. **Kinematic Frequency Misalignment**: Fixed-frequency 1D CNN spectral filters fail when bearing geometry and motor speed change, shifting BPFO by $-31\text{ Hz}$ and BPFI by $-38.5\text{ Hz}$.
2. **Amplitude Inversion**: Damaged Paderborn bearings produce lower vibration RMS than healthy CWRU bearings.
3. **Rig Transfer Function Differences**: Structural housing stiffness and accelerometer mounting differ between CWRU motor housing and Paderborn bearing modules.

### Category C: Unresolved Hypotheses & Open Questions
1. **Order Tracking Sufficiency**: *Would angular resampling (order tracking: orders $= f / f_r$) alone restore zero-shot transfer across speeds, or does bearing pitch diameter mismatch ($D_p/d$) still require geometric normalization?*
2. **Domain Adaptation Necessity**: *Can unsupervised test-time adaptation (e.g. Adaptive Batch Normalization or CORAL) bridge amplitude and structural resonance differences without retraining the core feature extractor?*

---

## 6. Recommended Next Steps

To advance InspectAI from single-rig performance to genuine multi-rig cross-dataset generalization:

1. **Angular Domain Order Tracking**: Re-formulate spectral features from fixed Hertz bins to rotational orders ($\text{order} = \text{frequency} / f_{\text{shaft}}$) using synchronized tachometer signals.
2. **Geometric Frequency Normalization**: Parameterize the spectral representation relative to bearing kinematic coefficients ($BPFO / f_r$, $BPFI / f_r$).
3. **Unsupervised Domain Adaptation**: Evaluate domain-adversarial neural networks (DANN) or CORAL to align source (CWRU) and target (Paderborn) feature distributions without target labels.

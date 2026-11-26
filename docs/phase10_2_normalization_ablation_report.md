# InspectAI: Normalization Strategy Ablation & Generalization Report (Phase 10.2)

## Executive Summary

Phase 10.1 hypothesized that the InspectAI 1D CNN's complete classification failure on unseen 0.014-inch race faults was caused by an "amplitude energy shortcut" under global normalization. To test this hypothesis scientifically, Phase 10.2 executed a controlled, leakage-safe ablation comparing three normalization strategies across identical 2-fold recording-level cross-validation on the development set (2,110 windows across 9 independent recordings).

**Key Empirical Finding**:
The amplitude-shortcut hypothesis was **refuted by empirical evidence**:
- Window-level instance normalization did **not** resolve the failure to classify severe Inner Race faults (`209.mat`), which remained 100% misclassified as Ball Fault.
- Worse, instance normalization **destroyed the model's ability to recognize minor Inner Race faults** (`105.mat`), dropping Inner Race F1 from **60.92% to 0.00%** because it forced 100% of `105` windows to be misclassified as Outer Race Fault.
- Amplitude variation is not a nuisance shortcut; in rolling element bearings, impact magnitude and load-zone modulation carry vital physical discriminative information.
- Global training-only normalization remains the superior, scientifically sound approach for the 1D CNN architecture on single-rig data.

---

## 1. Experimental Objective & Controlled Design

The objective was to isolate the effect of amplitude normalization on recording-level generalization without changing the architecture, optimizer, or random seeds.

### 1.1 Evaluated Normalization Strategies

1. **`global` (Baseline)**:
   Scalar mean and standard deviation fitted strictly on training recordings of each fold:
   $$\tilde{x}[n] = \frac{x[n] - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
   Preserves relative amplitude differences between quiet (Normal) and violent (Outer Race) vibration runs.

2. **`instance` (Window-Level Z-Score)**:
   Mean and standard deviation computed independently for each individual 2,048-sample window:
   $$\tilde{x}[n] = \frac{x[n] - \mu_w}{\max(\sigma_w, \epsilon)}$$
   Eliminates all amplitude/energy differences, forcing the network to classify solely on normalized temporal wave shape and frequency distribution. Zero-variance/constant windows are safely mapped to zero.

3. **`none` (Raw Physical Acceleration)**:
   Passthrough mode with no amplitude scaling after DC removal and 20–5,000 Hz Butterworth bandpass filtering:
   $$\tilde{x}[n] = x[n]$$
   Evaluates raw numerical stability and direct physical acceleration in $g$.

### 1.2 Strict Experimental Controls

To ensure an unconfounded comparison:
- **Partitions**: Identical 2-fold Stratified Group K-Fold on the 9 development recordings.
  - Fold 0 Train: `{"100", "105", "222", "234"}` (938 windows) | Val: `{"98", "99", "118", "130", "209"}` (1,172 windows).
  - Fold 1 Train: `{"98", "99", "118", "130", "209"}` (1,172 windows) | Val: `{"100", "105", "222", "234"}` (938 windows).
- **Holdout Test Isolation**: The previously evaluated holdout recordings (`97`, `169`, `185`, `197`) remained **100% untouched and isolated**.
- **Model Architecture**: Exact `VibrationCNN1D` (73,156 trainable parameters, Conv1D blocks with kernel sizes 15, 7, 3).
- **Hyperparameters**: AdamW ($lr=10^{-3}$, weight decay $=10^{-4}$), ReduceLROnPlateau ($patience=3, factor=0.5$), batch size 32, max epochs 30.
- **Early Stopping**: Identical patience of 7 epochs monitoring validation Macro-F1.
- **Hardware**: Single CPU execution.

---

## 2. Experimental Results & Performance Comparison

### 2.1 Pooled Out-of-Fold (OOF) Benchmark

Across all 2,110 development windows evaluated out-of-fold:

| Normalization Strategy | Pooled OOF Accuracy | Pooled OOF Macro-F1 | Normal F1 | Ball Fault F1 | Inner Race F1 | Outer Race F1 | Total CPU Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`global` (Baseline)** | **86.26%** | **83.78%** | **100.00%** | 79.56% | **60.92%** | **94.63%** | 44.2 s |
| **`instance` (Per-Window)** | 77.82% | 65.05% | **100.00%** | **80.03%** | **0.00%** | 80.17% | 59.7 s |
| **`none` (Raw Amplitudes)** | 85.55% | 83.11% | **100.00%** | **80.03%** | 60.44% | 91.97% | 56.4 s |

---

### 2.2 Confusion Matrices

#### A. Global Normalization (`global`)
```
                     Predicted
               Normal    Ball    Inner   Outer
True Normal     700        0       0       0
True Ball         0      471       0       0
True Inner        0      242     226       0
True Outer        0        0      48     423
```
- Total correct: 1,820 / 2,110 (86.26%)
- Inner Race Fault: 226 correct, 242 misclassified as Ball Fault.
- Outer Race Fault: 423 correct, 48 misclassified as Inner Race Fault.

#### B. Instance Normalization (`instance`)
```
                     Predicted
               Normal    Ball    Inner   Outer
True Normal     700        0       0       0
True Ball         0      471       0       0
True Inner        0      235       0     233
True Outer        0        0       0     471
```
- Total correct: 1,642 / 2,110 (77.82%)
- **Inner Race Fault: 0 correct out of 468 (0.00% recall)**.
- 235 Inner Race windows misclassified as Ball Fault; 233 misclassified as Outer Race Fault!

#### C. No Amplitude Normalization (`none`)
```
                     Predicted
               Normal    Ball    Inner   Outer
True Normal     700        0       0       0
True Ball         0      471       0       0
True Inner        0      235     233       0
True Outer        0        0      70     401
```
- Total correct: 1,805 / 2,110 (85.55%)
- Performance closely tracks `global` normalization.

---

### 2.3 Per-Recording Breakdown

Examining how each strategy performed across individual physical recordings:

| Recording | True Class | Defect Size | RMS (g) | Global Accuracy | Instance Accuracy | None Accuracy | Primary Instance Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **98** | Normal | 0.000" | 0.0664 | 100.00% | 100.00% | 100.00% | None |
| **99** | Normal | 0.000" | 0.0643 | 100.00% | 100.00% | 100.00% | None |
| **100** | Normal | 0.000" | 0.0659 | 100.00% | 100.00% | 100.00% | None |
| **118** | Ball Fault | 0.007" | 0.1392 | 100.00% | 100.00% | 100.00% | None |
| **222** | Ball Fault | 0.021" | 0.1356 | 100.00% | 100.00% | 100.00% | None |
| **105** | Inner Race | 0.007" | 0.2915 | **97.00%** | **0.00%** | **100.00%** | 100% predicted as Outer Race |
| **209** | Inner Race | 0.021" | 0.5254 | **0.00%** | **0.00%** | **0.00%** | 100% predicted as Ball Fault |
| **130** | Outer Race | 0.007" | 0.6695 | 100.00% | 100.00% | 100.00% | None |
| **234** | Outer Race | 0.021" | 0.5832 | 79.66% | **100.00%** | 70.34% | None |

---

## 3. Physical & Diagnostic Analysis

### 3.1 Why Did Instance Normalization Fail?

```
RAW ACCELEROMETER RECORDING:
  Inner Race 105 (0.007"): [Moderate RMS ~0.29g, Periodic Impulses with Shaft Modulation]
  Outer Race 130 (0.007"): [High RMS ~0.67g, Heavy Stationary Impact Spikes]
                              |
                              v
UNDER INSTANCE NORMALIZATION ((x - mu_w) / sigma_w):
  Both signals are forced to variance = 1.0.
  The amplitude distinction is erased.
  Without absolute scale, the 1D CNN cannot separate the 105 impulse shape from Outer Race!
  RESULT: 100% of Recording 105 windows assigned to Outer Race Fault.
```

1. **Destruction of Modulation Envelope Discrimination**:
   - In rolling element bearings, an outer race defect is stationary relative to the housing. Every ball impact occurs under a constant load-zone transfer function, creating sharp, uniform acoustic spikes.
   - An inner race defect rotates with the shaft. It passes through the heavy load zone and the light load zone once per shaft revolution ($f_r \approx 29.9\text{ Hz}$). The resulting vibration train is naturally amplitude-modulated.
   - When each 2,048-sample window (~170.7 ms, containing ~5 shaft rotations) is normalized to unit variance independently, the peak-to-background ratio is distorted. The CNN loses the amplitude contrast that identifies the inner race modulation envelope, causing it to confuse `105` with `130`.

2. **The 209 Failure is Not an Amplitude Artifact**:
   - Recording `209.mat` (0.021" Inner Race) was predicted 100% as Ball Fault under `global`, `instance`, AND `none`.
   - This proves that `209.mat`'s misclassification is **not** an artifact of global normalization scaling. Even with all amplitude information completely stripped away (`instance`), the network still mapped it to Ball Fault!
   - This indicates a representation limit in the temporal receptive field or kernel structure: severe 0.021-inch inner race spalls produce smeared acoustic resonance patterns that the 1D CNN's current filter bank maps to Ball Fault dynamics.

---

## 4. Limitations & Threats to Validity

1. **Dataset Scale**: The development set contains 9 recordings (2,110 windows). While leakage-safe at the recording level, each fold has only 1 or 2 recordings per fault class.
2. **Defect Severity vs. Kinematic Frequency**: In CWRU, defect sizes (0.007", 0.014", 0.021") introduce severe structural changes in the impact waveform that cannot be resolved solely by adjusting preprocessing normalization.

---

## 5. Scientific Recommendation & Decision

1. **Reject Window-Level Instance Normalization**:
   - Instance normalization severely degrades model performance (Macro-F1 drops from 83.78% to 65.05%) and completely destroys Inner Race detection. It must **not** replace the existing baseline normalizer.
2. **Retain Global Normalization as the Primary Preprocessing Standard**:
   - `global` normalization maintains the highest Macro-F1 (83.78%), the highest overall accuracy (86.26%), and the best balance of class F1 scores on the CWRU benchmark.
3. **Keep Normalization Configurable**:
   - Retain `strategy: "global" | "instance" | "none"` in `src/training/normalizer.py` so that external multi-rig datasets (e.g. Jiangnan University) with incompatible sensor gains can still select instance normalization if cross-sensor calibration is absent.
4. **Recommended Next Stage (Phase 10.3)**:
   - Investigate **architectural representation enhancements** (e.g., multiscale receptive fields or frequency-domain inputs via STFT/FFT) rather than preprocessing amplitude manipulation, to address the underlying kinematic classification failures.


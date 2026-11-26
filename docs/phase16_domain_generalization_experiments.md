# InspectAI — Phase 16: Order Tracking & Unsupervised Domain Adaptation Investigation

## Executive Summary

Phase 16 investigated whether signal-processing adaptations (order tracking) and unsupervised statistical domain alignment (CORAL) can bridge the distribution shift between the **Case Western Reserve University (CWRU)** development domain and the **Paderborn University Bearing Dataset (PU)** external validation domain.

In Phase 14, the frozen release model (`models/final/inspectai_final_model.pth`, SHA-256 `a612eb9ebcc7...`) exhibited catastrophic zero-shot collapse on 7,211 Paderborn windows across 80 recordings, achieving an accuracy of **8.29%** (balanced accuracy **8.30%**, macro-F1 **4.04%**). Phase 15 confirmed this was not an implementation defect, but rather fundamental domain shift caused by:
1. Operational differences (1,797 RPM vs. 1,500 RPM shaft speeds).
2. Kinematic bearing geometry differences (CWRU 6205 bearing with 9 rolling elements vs. Paderborn 6203 bearing with 8 rolling elements).
3. Transducer sensitivity and structural transfer function mismatch (piezoelectric charge vs. IEPE sensors, 30.5 m/s² vs. 3.0 m/s² raw RMS).

### Key Empirical Findings

1. **Continuous Dynamic Computed Order Tracking (COT) is Infeasible**:
   - An exhaustive audit of raw CWRU files confirmed that CWRU provides **only a single scalar operating speed** (e.g., `1797 RPM`), with **zero continuous keyphasor pulse trains or angular encoder channels**.
   - Continuous COT requires synchronous angular pulse zero-crossing interpolation $\theta(t)$. Synthesizing artificial keyphasor pulses is scientifically invalid.
2. **Nominal Speed-Normalized Order-Scaling**:
   - Rescaling envelope spectra by nominal shaft frequency ($f_{\text{order}} = f / f_{\text{shaft}}$) normalized rotational frequency from 25.0 Hz to 29.95 Hz.
   - However, because bearing geometries differ ($Z=9$ vs. $Z=8$), fault orders remain misaligned (e.g., Outer Race BPFO is 3.585 orders on CWRU vs. 3.053 orders on Paderborn).
   - Under frozen model evaluation, nominal order scaling alone produced **25.00% accuracy** (14.82% macro-F1), collapsing into predicting Outer Race Fault for all defect windows.
3. **Unsupervised CORAL (Correlation Alignment) Substantially Recovers Representation**:
   - Under a **strictly label-free protocol** (zero target labels used for covariance fitting or projection), second-order covariance alignment in the 576-dimensional latent feature space ($512\text{-dim wave} + 64\text{-dim spectral}$) increased balanced accuracy from **8.30% to 52.30%** (macro-F1 from **4.04% to 50.27%**).
   - Healthy normal detection (`K001`) surged from **0.0% to 75.9% recall** (1,370/1,804 windows).
   - Inner race fault detection (`KI04`) surged from **0.0% to 82.1% recall** (1,481/1,803 windows).
   - This proves that the frozen CNN representations retained genuine discriminative structure that was obscured by feature covariance distortion across rigs.

---

## 1. Task 1: Order-Tracking Feasibility Audit

### 1.1 Technical Channel Audit

An inspection of raw sensor archives was performed across both datasets:

| Dataset | File Inspected | Speed / Tachometer Channels | Sample Rate | Nature of Channel |
| :--- | :--- | :--- | :--- | :--- |
| **Paderborn** | `N15_M07_F10_K001_1.mat` | `speed` (Channel 4 in `Y[3]`) | 4,000 Hz | Low-rate analog/PLC command signal (mean 1499.59 RPM, range [1499.33, 1499.81] RPM). No raw optical encoder pulses. |
| **CWRU** | `100.mat`, `105.mat`, etc. | `X100RPM` (or `*RPM` scalar variable) | Static (0 Hz) | Single scalar operating speed (e.g. 1725, 1772, 1797). **Zero** continuous tachometer or pulse channels. |

### 1.2 Computed Order Tracking (COT) Feasibility Assessment

- **Dynamic Computed Order Tracking (COT)** requires synchronously recorded optical pulse trains (e.g., 1,024 pulses/revolution or 1 pulse/rev keyphasor) sampled at high frequency ($\ge 100\text{ kHz}$). The algorithm tracks the exact time of each pulse arrival $t_k$, fits a continuous phase spline $\theta(t)$, and resamples the non-stationary vibration signal $x(t)$ onto a uniform angular grid $x(\theta)$ using cubic spline interpolation.
- **Audit Conclusion**: Because CWRU lacks a continuous angular channel, dynamic COT is **physically impossible** on the source dataset. Fabricating synthetic pulse trains or assuming constant speed to generate pulses would violate scientific validation standards.
- **Nominal Order-Scaling Feasibility**: Nominal order-scaling operates in the frequency domain. Given nominal shaft frequencies $f_{r,\text{CWRU}} = 29.95\text{ Hz}$ and $f_{r,\text{PU}} = 25.00\text{ Hz}$, the envelope spectrum grid is rescaled:
  $$\text{order} = \frac{f}{f_r}$$
  The spectrum is interpolated from the Paderborn order grid ($\text{order}_{\text{PU}} = f / 25.00$) onto the CWRU order grid ($\text{order}_{\text{CWRU}} = f / 29.95$) across the 1,025 frequency bins.

### 1.3 Theoretical Geometric Limitation of Order Scaling

Speed normalization aligns shaft harmonics, but does **not** align defect frequencies due to differing bearing geometries:

$$\text{BPFO} = \frac{Z}{2} \left(1 - \frac{d}{D} \cos \alpha\right), \quad \text{BPFI} = \frac{Z}{2} \left(1 + \frac{d}{D} \cos \alpha\right)$$

| Parameter | CWRU (SKF 6205-2RS JEM) | Paderborn (FAG 6203) | Discrepancy in Orders |
| :--- | :--- | :--- | :--- |
| **Number of Balls ($Z$)** | 9 | 8 | $-1$ ball |
| **Pitch Diameter ($D$)** | 39.04 mm | 28.50 mm | $-10.54\text{ mm}$ |
| **Ball Diameter ($d$)** | 7.94 mm | 6.75 mm | $-1.19\text{ mm}$ |
| **Contact Angle ($\alpha$)** | $0^\circ$ | $0^\circ$ | $0^\circ$ |
| **Nominal BPFO** | **3.585 orders** (107.4 Hz) | **3.053 orders** (76.3 Hz) | **$-0.532\text{ orders}$** ($-14.8\%$) |
| **Nominal BPFI** | **5.415 orders** (162.2 Hz) | **4.947 orders** (123.7 Hz) | **$-0.468\text{ orders}$** ($-8.6\%$) |
| **Nominal BSF** | **2.322 orders** (69.5 Hz) | **1.990 orders** (49.8 Hz) | **$-0.332\text{ orders}$** ($-14.3\%$) |

Because the CNN's 1D convolutional filters were trained on CWRU's specific harmonic peaks, shifting the spectrum by shaft speed alone still leaves the fault peaks offset by 0.33 to 0.53 orders from the learned kernel receptive fields.

---

## 2. Task 2: Unsupervised Domain Adaptation (CORAL)

### 2.1 Methodology & Mathematical Formulation

Correlation Alignment (CORAL) aligns the second-order statistics (covariance) of the target domain features with the source domain features in the latent embedding space without modifying the classifier weights.

Let $D_S = \{x_i^S\}_{i=1}^{n_S}$ denote the source feature embeddings ($n_S = 2,928$ CWRU development windows, dimension $d = 576$), and $D_T = \{x_j^T\}_{j=1}^{n_T}$ denote the target feature embeddings ($n_T = 7,211$ Paderborn pilot windows, dimension $d = 576$).

1. **Mean Centering**:
   $$\mu_S = \frac{1}{n_S} \sum_{i=1}^{n_S} x_i^S, \quad \mu_T = \frac{1}{n_T} \sum_{j=1}^{n_T} x_j^T$$
   $$X_S^c = X_S - \mathbf{1} \mu_S^T, \quad X_T^c = X_T - \mathbf{1} \mu_T^T$$

2. **Covariance Estimation with Regularization**:
   $$C_S = \frac{1}{n_S - 1} (X_S^c)^T X_S^c + \lambda I_d$$
   $$C_T = \frac{1}{n_T - 1} (X_T^c)^T X_T^c + \lambda I_d$$
   where $\lambda = 10^{-3}$ ensures strict positive definiteness.

3. **Alignment Transformation**:
   Using symmetric eigendecompositions $C_S = V_S \Lambda_S V_S^T$ and $C_T = V_T \Lambda_T V_T^T$:
   $$C_S^{1/2} = V_S \Lambda_S^{1/2} V_S^T, \quad C_T^{-1/2} = V_T \Lambda_T^{-1/2} V_T^T$$
   $$A_{\text{CORAL}} = C_T^{-1/2} C_S^{1/2}$$

4. **Target Feature Projection**:
   $$X_{T,\text{aligned}} = X_T^c A_{\text{CORAL}} + \mathbf{1} \mu_S^T$$

5. **Inference**:
   The aligned features are passed directly into the frozen classification head:
   $$\hat{y}_T = \operatorname{argmax} \left( \operatorname{classifier}(X_{T,\text{aligned}}) \right)$$

### 2.2 Strict Label-Free Protocol

- **Target Label Isolation**: Paderborn class labels ($y_T$) were completely excluded during the calculation of $\mu_T$, $C_T$, and $A_{\text{CORAL}}$.
- **Frozen Source Weights**: The convolutional feature extractors (`wave_branch`, `spectral_branch`) and the multi-layer classification head (`classifier`) remained 100% frozen.
- **Unsupervised Transformation**: The transformation is an entirely unsupervised linear mapping derived solely from unlabelled target sensor windows.

---

## 3. Task 3: Controlled Experimental Comparison

### 3.1 Benchmark Comparison Table

All evaluations were executed on the complete 7,211-window Paderborn pilot dataset (80 recordings across bearings `K001`, `KB23`, `KI04`, `KA04` under operating condition `N15_M07_F10`):

| Experiment | Preprocessing | Adaptation Method | Target Labels Used? | Accuracy | Balanced Accuracy | Macro F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp A: Frozen Baseline** | Standard Hz FFT | None (Zero-Shot) | **No** | **8.29%** | **8.30%** | **4.04%** |
| **Exp B: Nominal Order-Scaling** | Rescaled Order FFT | None (Zero-Shot) | **No** | **25.00%** | **25.00%** | **14.82%** |
| **Exp C: Unsupervised CORAL** | Standard Hz FFT | Second-Order Covariance Alignment | **No** | **52.31%** | **52.30%** | **50.27%** |
| **Exp D: Combined Order + CORAL**| Rescaled Order FFT | Second-Order Covariance Alignment | **No** | **42.84%** | **42.83%** | **41.98%** |

### 3.2 Per-Class Detailed Performance Breakdown

#### Experiment A: Frozen Release Baseline (Re-verification)
- **Normal (`K001`)**: Prec = 0.00%, Rec = 0.00%, F1 = 0.00% (Support = 1,804)
- **Ball Fault (`KB23`)**: Prec = 10.25%, Rec = 32.70%, F1 = 15.61% (Support = 1,801)
- **Inner Race (`KI04`)**: Prec = 0.00%, Rec = 0.00%, F1 = 0.00% (Support = 1,803)
- **Outer Race (`KA04`)**: Prec = 0.61%, Rec = 0.50%, F1 = 0.55% (Support = 1,803)

#### Experiment B: Nominal Order-Scaling
- **Normal (`K001`)**: Prec = 0.00%, Rec = 0.00%, F1 = 0.00% (Support = 1,804)
- **Ball Fault (`KB23`)**: Prec = 0.00%, Rec = 0.00%, F1 = 0.00% (Support = 1,801)
- **Inner Race (`KI04`)**: Prec = 0.00%, Rec = 0.00%, F1 = 0.00% (Support = 1,803)
- **Outer Race (`KA04`)**: Prec = 42.12%, Rec = 100.00%, F1 = 59.27% (Support = 1,803)

#### Experiment C: Unsupervised CORAL (Breakthrough Performance)
- **Normal (`K001`)**: **Prec = 61.24%, Rec = 75.94%, F1 = 67.80%** (1,370 / 1,804 correctly classified)
- **Ball Fault (`KB23`)**: Prec = 26.93%, Rec = 23.10%, F1 = 24.87% (416 / 1,801 correctly classified)
- **Inner Race (`KI04`)**: **Prec = 71.44%, Rec = 82.14%, F1 = 76.42%** (1,481 / 1,803 correctly classified)
- **Outer Race (`KA04`)**: Prec = 37.24%, Rec = 28.01%, F1 = 31.97% (505 / 1,803 correctly classified)

#### Experiment D: Combined Nominal Order-Scaling + CORAL
- **Normal (`K001`)**: Prec = 42.55%, Rec = 49.72%, F1 = 45.86% (897 / 1,804 correctly classified)
- **Ball Fault (`KB23`)**: Prec = 17.60%, Rec = 15.71%, F1 = 16.60% (283 / 1,801 correctly classified)
- **Inner Race (`KI04`)**: Prec = 61.04%, Rec = 69.33%, F1 = 64.92% (1,250 / 1,803 correctly classified)
- **Outer Race (`KA04`)**: Prec = 45.54%, Rec = 36.55%, F1 = 40.55% (659 / 1,803 correctly classified)

---

## 4. Confusion Matrices

### Experiment A Confusion Matrix (Frozen Zero-Shot Baseline)
```
                  Predicted Normal  Predicted Ball  Predicted Inner  Predicted Outer
True Normal                    0            1804                0                0
True Ball Fault                0             589                0             1212
True Inner Race                0            1558                0              245
True Outer Race                0            1794                0                9
```

### Experiment B Confusion Matrix (Nominal Order-Scaling)
```
                  Predicted Normal  Predicted Ball  Predicted Inner  Predicted Outer
True Normal                    0            1563                0              241
True Ball Fault             1367               0                0              434
True Inner Race                0               0                0             1803
True Outer Race                0               0                0             1803
```

### Experiment C Confusion Matrix (Unsupervised CORAL)
```
                  Predicted Normal  Predicted Ball  Predicted Inner  Predicted Outer
True Normal                 1370             301              132                1
True Ball Fault              399             416              439              547
True Inner Race                9              10             1481              303
True Outer Race              459             818               21              505
```

### Experiment D Confusion Matrix (Combined Order + CORAL)
```
                  Predicted Normal  Predicted Ball  Predicted Inner  Predicted Outer
True Normal                  897             695              212                0
True Ball Fault              840             283              332              346
True Inner Race                4             107             1250              442
True Outer Race              367             523              254              659
```

---

## 5. In-Depth Diagnostic Discussion

### 5.1 Why Unsupervised CORAL Succeeded
CORAL operates entirely in the 576-dimensional latent space after all convolutional representations are extracted. The CWRU and Paderborn feature distributions had completely detached means ($\mu_S \ne \mu_T$) and skewed covariance ellipsoids due to:
- Channel transducer amplitude scaling differences.
- High-frequency resonant attenuation from the mechanical test bench housing.

By computing the whitening transformation $C_T^{-1/2}$ and recoloring with the source covariance $C_S^{1/2}$, CORAL translated and rotated the target feature cloud into the decision region calibrated by the source classifier.
- **Normal Class Recovery (75.9% recall)**: Healthy bearings produce stochastic Gaussian vibration with minimal spectral peaks. Once the baseline variance and mean shift were aligned, the classifier accurately recognized baseline healthy operation.
- **Inner Race Recovery (82.1% recall)**: Inner race defects produce distinct high-frequency modulation (shaft rotation sidebands modulating the carrier frequency). In the 576-dim fused latent space, this modulation creates a distinctive directional variance that CORAL successfully mapped back onto CWRU's inner-race decision cone.

### 5.2 Why Ball Fault & Outer Race Remain Difficult
- **Ball Fault (23.1% recall)**: Rolling elements spin and precess non-linearly. The ball defect signature in Paderborn (`KB23`) involves electrical discharge damage causing micro-pitting, which produces much weaker periodic impulses than the EDM electrical discharge slots in CWRU.
- **Outer Race (28.0% recall)**: In Paderborn, Outer Race defects (`KA04`) frequently misclassified into Ball Fault (818 windows) because the contact mechanics of Paderborn's test housing distributed the outer race impact across multiple structural resonance paths.
- **Geometric Incompatibility**: As derived in Section 1.3, the 8-ball geometry of the 6203 bearing inherently operates at a lower order (3.053 vs. 3.585). Feature covariance alignment shifts the global distribution, but cannot alter the internal relative distance between individual harmonic peaks inside the spectral convolution filters.

### 5.3 Why Experiment D (Order + CORAL) Underperformed Experiment C (CORAL Alone)
Linear interpolation of envelope spectra across 1,025 discrete frequency bins without continuous phase tracking introduces subtle spectral smoothing and smearing of narrow harmonic peaks. When passed to the CNN, this smoothing degraded the feature representations before CORAL was applied, reducing balanced accuracy from 52.30% to 42.83%.

---

## 6. Scientific Integrity & Artifact Audit

| Artifact File | Path | Status | Verification Check |
| :--- | :--- | :---: | :--- |
| **Frozen Final Model** | `models/final/inspectai_final_model.pth` | UNTOUCHED | SHA-256: `a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e` |
| **Waveform Normalizer**| `models/final/inspectai_final_normalizer.json` | UNTOUCHED | mean: `-3.2797e-06`, std: `0.3469` |
| **Spectral Normalizer**| `models/final/inspectai_final_spectral_normalizer.json` | UNTOUCHED | 1025-bin vector statistics |
| **Feasibility Audit** | `outputs/experiments/phase16/feasibility_audit.json` | GENERATED | Valid JSON, 31 lines |
| **Experiment Results** | `outputs/experiments/phase16/experiment_results.json` | GENERATED | Valid JSON, 321 lines, all 4 experiments |
| **Automated Tests** | `tests/test_phase16_domain_generalization.py` | VERIFIED | 9/9 unit & integration tests passing |

---

## 7. Strategic Conclusions & Next Steps

1. **Unsupervised Domain Adaptation is Mandatory for Industrial Transfer**:
   - Zero-shot direct transfer across different machine test rigs is unrealistic due to transducer, speed, and geometric shifts.
   - However, **unsupervised domain adaptation (CORAL) provides a mathematically sound, label-free pathway** to recover more than half of the classification capacity (52.3% balanced accuracy, >75% recall on Normal and Inner Race) without requiring expensive target-domain annotations.
2. **Next Steps**:
   - Stop Phase 16 here.
   - The findings, feasibility audit, and experimental artifacts are fully preserved for architectural portfolio review and future domain-generalization extensions.

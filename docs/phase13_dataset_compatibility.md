# InspectAI – Phase 13: Dataset Compatibility Assessment & Kinematic Domain Analysis

**Date**: November 2026  
**Auditor**: InspectAI Release & Validation Engineering  
**Scope**: Technical comparison between InspectAI requirements and candidate external bearing datasets.

---

## 1. InspectAI Technical Baseline Specification

InspectAI's frozen candidate architecture, **`SpectralFusionCNN`**, was trained on conditioned vibration signals from the CWRU Bearing Data Center. Its inference contract requires:

| Property | InspectAI Requirement | Baseline Provenance (CWRU) |
|---|---|---|
| **Architecture** | `SpectralFusionCNN` | Dual-branch 1D CNN (89,604 trainable parameters) |
| **Time-Domain Input** | Array of shape $(1, 1, 2048)$ | 2,048 samples ($\approx 170.67\text{ ms}$ at 12 kHz) |
| **Spectral-Domain Input** | Array of shape $(1, 1, 1025)$ | Positive bins ($0 - 6000\text{ Hz}$) of zero-centered Hilbert envelope spectrum ($N_{\text{fft}} = 2048$) |
| **Sampling Frequency ($f_s$)** | Exactly **12,000 Hz** | Resampled from 48 kHz (Normal) or native 12 kHz (Faults) |
| **Output Classes ($C=4$)** | Normal (0), Ball (1), Inner Race (2), Outer Race (3) | Mutually exclusive single-label classification |
| **Operating Shaft Speed ($f_r$)**| $1720 - 1797\text{ RPM}$ ($28.7 - 29.95\text{ Hz}$) | Induction motor under 0, 1, 2, and 3 HP loads |
| **Bearing Geometry** | SKF 6205-2RS JEM ($Z=9, d=7.94\text{ mm}, D_p=39.04\text{ mm}$) | Deep-groove ball bearing |
| **Sensor Modality** | Accelerometer | Mounted vertically at 12 o'clock on drive-end housing |
| **Signal Units** | Physical acceleration in $g$ ($1g \approx 9.81\text{ m/s}^2$) | Accelerometer voltage calibrated to $g$ |

---

## 2. In-Depth Cross-Dataset Compatibility Matrix

| Evaluation Dimension | InspectAI Baseline | Paderborn University (PU) | MFPT Dataset | Southeast Univ (SEU) | Jiangnan Univ (JNU) |
|---|---|---|---|---|---|
| **Native Sampling Rate** | 12,000 Hz | 64,000 Hz | 48,828 / 97,656 Hz | **5,120 Hz** | 50,000 Hz |
| **Resampling Feasibility** | Baseline | **Feasible** ($f_s \to 12\text{ kHz}$ via factor $3/16$) | **Feasible** ($48828 \to 12000$ via $1000/4069$) | **Incompatible** ($5.12 < 12\text{ kHz}$) | **Feasible** ($50000 \to 12000$ via $6/25$) |
| **Nyquist Bandwidth** | 6,000 Hz | 32,000 Hz (Excess bandwidth preserved) | 24,414 Hz (Excess preserved) | **2,560 Hz** (Band-limited void) | 25,000 Hz (Excess preserved) |
| **Rotational Speed ($f_r$)**| 1720–1797 RPM (29.5 Hz) | 1500 RPM (25.0 Hz) & 900 RPM (15.0 Hz) | 1000 RPM (16.67 Hz) | 1200 & 1800 RPM | 600, 800, 1000 RPM |
| **Speed Shift Ratio vs Baseline**| 1.00x | **0.85x** (1500 RPM) / 0.51x (900 RPM) | **0.56x** (1000 RPM) | 0.61x (1200 RPM) / 1.02x (1800 RPM) | 0.34x – 0.56x |
| **Bearing Geometry** | SKF 6205 ($Z=9$) | **SKF 6203 ($Z=8$)** | **Nice #1012 ($Z=8$)** | ER-16K ($Z=9$) | Unspecified |
| **Pitch Diameter ($D_p$)** | 39.04 mm | 28.50 mm | 31.62 mm (1.245 in) | 38.50 mm | Unspecified |
| **Ball Diameter ($d$)** | 7.94 mm | 6.75 mm | 5.97 mm (0.235 in) | 7.94 mm | Unspecified |
| **Ratio ($d / D_p$)** | 0.2034 | 0.2368 | 0.1888 | 0.2062 | Unspecified |
| **Theoretical BPFO Order** | 3.585 orders | 3.053 orders (-14.8%) | 3.245 orders (-9.5%) | 3.572 orders (-0.4%) | Unspecified |
| **Theoretical BPFI Order** | 5.415 orders | 4.947 orders (-8.6%) | 4.755 orders (-12.2%) | 5.428 orders (+0.2%) | Unspecified |
| **Theoretical BSF Order** | 2.357 orders | 1.993 orders (-15.4%) | 2.551 orders (+8.2%) | 2.321 orders (-1.5%) | Unspecified |
| **Nominal BPFO Frequency** | **105.8 Hz** | **76.3 Hz** (-27.9%) | **54.1 Hz** (-48.9%) | 64.3 Hz / 107.2 Hz | ~35 – 59 Hz |
| **Nominal BPFI Frequency** | **159.8 Hz** | **123.7 Hz** (-22.6%) | **79.3 Hz** (-50.4%) | 97.7 Hz / 162.8 Hz | ~53 – 90 Hz |
| **Nominal BSF Frequency** | **69.5 Hz** | **49.8 Hz** (-28.3%) | **42.5 Hz** (-38.8%) | 41.8 Hz / 69.6 Hz | ~23 – 39 Hz |
| **Normal Class Support** | Full | **Full (6 Bearings)** | **Full (3 Files)** | Full | Full |
| **Inner Race Support** | Full | **Full (12 Bearings)** | **Full (7 Files)** | Full | Full |
| **Outer Race Support** | Full | **Full (6 Bearings)** | **Full (7 Files)** | Full | Full |
| **Ball Fault Support** | Full | **Full (3 Bearings: KB23, 24, 27)** | **STRICTLY ABSENT** | Full | Full |
| **Overall Compatibility** | Reference | **HIGH (Strongest Candidate)** | **MODERATE (Restricted 3-Class)** | **REJECTED (Sampling Rate)** | **REJECTED (Licensing/Hosting)**|

---

## 3. Kinematic Domain Shift & Harmonic Impact Analysis

### 3.1 Characteristic Bearing Frequency Physics
Bearing fault diagnostic theory dictates that rolling elements passing over localized raceway defects generate impulsive acoustic bursts at periodic intervals determined by shaft angular velocity $f_r$ and bearing pitch/ball dimensions:

$$\text{BPFO} = \frac{Z}{2} f_r \left(1 - \frac{d}{D_p}\cos\alpha\right)$$
$$\text{BPFI} = \frac{Z}{2} f_r \left(1 + \frac{d}{D_p}\cos\alpha\right)$$
$$\text{BSF} = \frac{D_p}{2d} f_r \left(1 - \left(\frac{d}{D_p}\cos\alpha\right)^2\right)$$

### 3.2 Impact Yield per Temporal Window ($L=2048$ samples $\approx 170.67\text{ ms}$)
In a 2,048-sample window at 12 kHz, the number of discrete impact bursts visible to the temporal convolutional filters varies dramatically across datasets:

| Fault State | CWRU (Baseline @ 1770 RPM) | Paderborn (PU @ 1500 RPM) | MFPT (@ 1000 RPM) |
|---|---|---|---|
| **Outer Race (BPFO)** | **18.1 impacts / window** | **13.0 impacts / window** | **9.2 impacts / window** |
| **Inner Race (BPFI)** | **27.3 impacts / window** | **21.1 impacts / window** | **13.5 impacts / window** |
| **Ball Fault (BSF)** | **11.9 impacts / window** | **8.5 impacts / window** | **N/A (No Ball Fault)** |

### 3.3 Frequency-Domain Shift in Hilbert Envelope Spectrum
InspectAI's spectral branch processes $K=1025$ positive frequency bins spanning $0\text{ Hz}$ to $6,000\text{ Hz}$ ($\Delta f = \frac{12000}{2048} \approx 5.859\text{ Hz/bin}$):

- **Outer Race Defect Peak (BPFO)**:
  - CWRU: Falls into **bin 18** ($105.8\text{ Hz} \pm 2.9\text{ Hz}$).
  - Paderborn (PU): Falls into **bin 13** ($76.3\text{ Hz} \pm 2.9\text{ Hz}$).
  - MFPT: Falls into **bin 9** ($54.1\text{ Hz} \pm 2.9\text{ Hz}$).
- **Inner Race Defect Peak (BPFI)**:
  - CWRU: Falls into **bin 27** ($159.8\text{ Hz} \pm 2.9\text{ Hz}$).
  - Paderborn (PU): Falls into **bin 21** ($123.7\text{ Hz} \pm 2.9\text{ Hz}$).
  - MFPT: Falls into **bin 14** ($79.3\text{ Hz} \pm 2.9\text{ Hz}$).

### 3.4 Key Scientific Hypothesis for Zero-Shot Transfer
Because the convolutional layers possess finite receptive fields, a model that memorized absolute frequency bin locations or fixed impact counts will misclassify external datasets. Conversely, if the multiscale convolutional filters learned invariant envelope modulation representations (such as harmonic comb spacing and transient decay profiles), the model will retain discriminative sensitivity across moderate speed shifts (e.g. CWRU 1770 RPM $\to$ Paderborn 1500 RPM, a 15% speed change).

---

## 4. Rejection Rationales for Incompatible Datasets

### 4.1 Southeast University (SEU): Disqualified on Bandwidth Deficit
- **Root Cause**: Acquisition sampling frequency is $5,120\text{ Hz}$.
- **Technical Barrier**: The Nyquist cutoff is $2,560\text{ Hz}$. Upsampling by interpolation to $12\text{ kHz}$ does not generate true physical high-frequency acoustic data. Over 57% of InspectAI's envelope spectrum ($2,560 - 6,000\text{ Hz}$) would consist of zero-energy voids or interpolation artifacts.
- **Scientific Decision**: Disqualified. Cross-dataset validation on artificially upsampled signals is physically invalid.

### 4.2 Jiangnan University (JNU): Disqualified on Governance & Hosting
- **Root Cause**: No persistent institutional DOI or permanent university repository; original lab servers are offline; distributed predominantly via unverified personal GitHub forks.
- **Technical Barrier**: Inability to verify authoritative metadata, sensor calibration constants, or legal distribution rights.
- **Scientific Decision**: Disqualified on data provenance and licensing reproducibility grounds.

### 4.3 University of Cincinnati (IMS): Disqualified on Task Mismatch
- **Root Cause**: Run-to-failure continuous test with four simultaneous bearings sharing a single shaft.
- **Technical Barrier**: High cross-talk between multiple decaying bearings; continuous lifetime degradation rather than discrete stationary fault conditions.
- **Scientific Decision**: Disqualified for steady-state 4-class fault classification.

---

## 5. Candidate Selection Conclusion

- **Primary Selected Dataset**: **Paderborn University (PU) Bearing Dataset (KAt)**.
  - Meets all 7 selection criteria: independent German university rig, CC BY-NC 4.0 license, SKF 6203 deep-groove ball bearing, 64 kHz high-fidelity sampling, 1500 RPM operating speed, comprehensive metadata, and support for all four InspectAI classes.
- **Secondary Benchmark Candidate**: **MFPT Bearing Dataset**.
  - Serves as a lightweight, independent 3-class restricted benchmark (Normal, Inner Race, Outer Race) under open Figshare DOI licensing.


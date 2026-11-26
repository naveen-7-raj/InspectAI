# InspectAI: Independent Dataset Research & Generalization Audit (Phase 10)

## Executive Summary

InspectAI was developed and evaluated using authentic accelerometer recordings from the Case Western Reserve University (CWRU) Bearing Data Center. Under an uncompromising recording-level partition, the frozen 1D CNN achieved **39.73% accuracy** and **34.04% Macro-F1** on four untouched holdout recordings. While the model exhibited high fidelity on Normal bearings (90.27% F1) and Ball Faults (45.88% F1), it completely failed on unseen 0.014-inch Inner Race and Outer Race faults (0.00% F1), misclassifying 100% of these windows as Ball Fault.

This document establishes the scientific audit of current failure modes, evaluates candidate independent bearing vibration datasets, analyzes signal compatibility, and formulates a leakage-safe external validation protocol.

---

## 1. Audit of Current Failure Modes

### 1.1 Empirical Observations (Verified Ground Truth)

The development and holdout splits were defined strictly at the recording level:

| Recording | Source File | Fault Class | Defect Diameter | Motor Load | Motor RPM | Split | Windows | RMS (g) | Peak (g) | Kurtosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **98** | `98.mat` | Normal | 0.000" | 1 HP | 1772 | **Dev** | 233 | 0.0664 | 0.3459 | -0.07 |
| **99** | `99.mat` | Normal | 0.000" | 2 HP | 1750 | **Dev** | 233 | 0.0643 | 0.3592 | -0.07 |
| **100** | `100.mat` | Normal | 0.000" | 3 HP | 1725 | **Dev** | 234 | 0.0659 | 0.3065 | -0.04 |
| **118** | `118.mat` | Ball Fault | 0.007" | 0 HP | 1797 | **Dev** | 236 | 0.1392 | 0.6070 | -0.02 |
| **222** | `222.mat` | Ball Fault | 0.021" | 0 HP | 1796 | **Dev** | 235 | 0.1356 | 1.6596 | 5.55 |
| **105** | `105.mat` | Inner Race Fault | 0.007" | 0 HP | 1797 | **Dev** | 233 | 0.2915 | 1.7390 | 2.40 |
| **209** | `209.mat` | Inner Race Fault | 0.021" | 0 HP | 1797 | **Dev** | 235 | 0.5254 | 3.7880 | 4.45 |
| **130** | `130.mat` | Outer Race Fault | 0.007" | 0 HP | 1797 | **Dev** | 235 | 0.6695 | 3.6304 | 4.65 |
| **234** | `234.mat` | Outer Race Fault | 0.021" | 0 HP | 1797 | **Dev** | 236 | 0.5832 | 6.6529 | 18.01 |
| **97** | `97.mat` | Normal | 0.000" | 0 HP | 1797 | **Test** | 116 | 0.0738 | 0.3113 | -0.24 |
| **185** | `185.mat` | Ball Fault | 0.014" | 0 HP | 1797 | **Test** | 234 | 0.1527 | 2.2782 | 14.77 |
| **169** | `169.mat` | Inner Race Fault | 0.014" | 0 HP | 1797 | **Test** | 234 | 0.1978 | 2.0108 | 18.96 |
| **197** | `197.mat` | Outer Race Fault | 0.014" | 0 HP | 1797 | **Test** | 234 | 0.1007 | 0.5506 | 0.06 |

#### Final Holdout Evaluation Metrics

- **Total Test Windows**: 818
- **Overall Accuracy**: 325 / 818 = 39.73%
- **Macro-F1**: 34.04%
- **Confusion Matrix**:
  - `True Normal` (116): 116 Pred Normal, 0 Pred Ball, 0 Pred Inner, 0 Pred Outer
  - `True Ball Fault` (234): 25 Pred Normal, 209 Pred Ball, 0 Pred Inner, 0 Pred Outer
  - `True Inner Race Fault` (234): 0 Pred Normal, 234 Pred Ball, 0 Pred Inner, 0 Pred Outer
  - `True Outer Race Fault` (234): 0 Pred Normal, 234 Pred Ball, 0 Pred Inner, 0 Pred Outer
- **Class F1 Scores**:
  - Normal: **90.27%** (Precision: 82.27%, Recall: 100.0%)
  - Ball Fault: **45.88%** (Precision: 30.87%, Recall: 89.32%)
  - Inner Race Fault: **0.00%** (Precision: 0.0%, Recall: 0.0%)
  - Outer Race Fault: **0.00%** (Precision: 0.0%, Recall: 0.0%)

### 1.2 Analysis of Failure Causes (Hypotheses vs. Evidence)

```
+---------------------------------------------------------------------------------------------------+
| VERIFIED OBSERVATION: All fault test windows originate from 0.014-inch defect recordings.          |
| In contrast, all training fault windows originate from 0.007-inch and 0.021-inch defect sizes.   |
+---------------------------------------------------------------------------------------------------+
                                              |
        +-------------------------------------+-------------------------------------+
        |                                                                           |
        v                                                                           v
[Hypothesis 1: Energy & Amplitude Bias]                     [Hypothesis 2: Defect Dynamic Shift]
Global z-score normalization preserves                      At 0.014", the ball entry-impact
absolute energy. Test Outer Race (RMS=0.1007g)              acoustic transients may differ
and Inner Race (RMS=0.1978g) overlap the                    from both minor pits (0.007") and
Ball Fault energy regime (RMS=0.135-0.153g).                severe spalls (0.021").
The model used energy as a shortcut feature.                
```

1. **Amplitude & Energy Bias (Supported by Signal Statistics)**:
   - *Observation*: The dev Outer Race recordings exhibited massive vibration energy (RMS of 0.6695 g and 0.5832 g). The dev Inner Race recordings exhibited RMS of 0.2915 g and 0.5254 g.
   - *Observation*: In contrast, holdout Outer Race recording `197.mat` exhibited an RMS of only **0.1007 g** (lower than Ball Fault), and `169.mat` exhibited an RMS of **0.1978 g**.
   - *Hypothesis*: The training normalizer applied a global mean and standard deviation (`std = 0.3643`), preserving relative waveform amplitudes. The 1D CNN likely learned to associate high-amplitude envelopes with race faults and moderate amplitudes with Ball Fault. When presented with low-amplitude race faults in the holdout set, the network default-classified them into the moderate-amplitude Ball Fault cluster.

2. **Defect Geometry & Load Zone Dynamics (Phenomenological Hypothesis)**:
   - *Observation*: In CWRU, Outer Race defects were cut at different radial locations relative to the load zone. Recording `197.mat` exhibits low peak amplitude (0.5506 g) and low kurtosis (0.06), indicating that the defect did not produce the pronounced shock pulses seen in `130.mat` (peak 3.6304 g) or `234.mat` (peak 6.6529 g).
   - *Hypothesis*: At 0 HP (no motor load), the radial contact force between rolling elements and the outer race is minimal. If the defect is outside the primary load zone or if clearance prevents sharp impact, the acoustic shock train is suppressed, masking characteristic BPFO frequencies.

3. **Sparse Training Set Diversity (Architectural Observation)**:
   - *Observation*: The training set included only two distinct physical recordings per fault class.
   - *Hypothesis*: With $N=2$ recordings per fault category, the convolutional filters cannot disentangle the invariant kinematic fault impulse frequencies (BPFO, BPFI, BSF) from the specific structural transfer function of the individual test runs. The network memorized recording-specific acoustic signatures rather than generalizable defect dynamics.

---

## 2. Independent Vibration Datasets Research

To establish whether the failure is an artifact of CWRU or reflects deep learning generalization limits, we researched independent, publicly available bearing vibration datasets:

### 2.1 Paderborn University (PU) Bearing Dataset
- **Institution**: Chair of Design and Drive Technology (KAt), Paderborn University, Germany.
- **Availability & License**: Publicly accessible via DataCite DOI for non-commercial academic research.
- **Structure**: 32 deep groove ball bearings (type 6203):
  - 6 Healthy bearings ($K001 - K006$)
  - 12 Artificially damaged bearings (EDM, drilling)
  - 14 Naturally degraded bearings from accelerated lifetime fatigue tests
- **Sampling Rate**: 64 kHz (piezoelectric accelerometer) synchronized with motor current, torque, and speed.
- **Operating Conditions**: 4 defined load-speed combinations (e.g. 1500 RPM, 0.7 Nm, 1000 N radial load).
- **Critical Taxonomy Constraint**: **No isolated Ball Fault / rolling element fault class.** The dataset categorizes damage into Healthy, Inner Ring (KI), Outer Ring (KA), and Combined Ring (KB).
- **Assessment**: Excellent for fatigue vs. artificial defect domain shift, but incompatible with InspectAI's 4-class taxonomy without collapsing into a 3-class problem.

### 2.2 MFPT (Machinery Failure Prevention Technology) Dataset
- **Institution**: Society for Machinery Failure Prevention Technology (hosted on NASA Prognostics Repository).
- **Availability & License**: Publicly accessible, open academic distribution.
- **Structure**: 23 total test runs:
  - 3 Baseline (Good/Healthy) conditions under 270 lbs load
  - 10 Outer Race fault conditions (3 under 270 lbs, 7 under 0–300 lbs load)
  - 7 Inner Race fault conditions (under 0–300 lbs load)
  - 3 Real-world machine recordings (uncontrolled industrial gearboxes)
- **Sampling Rate**: 48,828 Hz (Normal, Outer Race) and 97,656 Hz (Inner Race).
- **Critical Taxonomy Constraint**: **No Ball Fault class.** The benchmark test-rig data includes only Normal, Inner Race, and Outer Race faults.
- **Assessment**: Incompatible with a 4-class single-model evaluation. Requires retraining as a 3-class classifier.

### 2.3 NASA IMS (Intelligent Maintenance Systems) Bearing Dataset
- **Institution**: Center for Intelligent Maintenance Systems, University of Cincinnati.
- **Availability & License**: Public open domain via NASA Prognostics Center of Excellence.
- **Structure**: Run-to-failure continuous lifetime tests:
  - 4 Rexnord ZA-2115 double-row bearings run continuously until failure (3 distinct endurance campaigns).
  - Test 1: Bearing 3 inner race failure, Bearing 4 roller element failure.
  - Test 2: Bearing 1 outer race failure.
  - Test 3: Bearing 3 outer race failure.
- **Sampling Rate**: 20 kHz (1-second snapshot = 20,480 samples every 10 or 20 minutes).
- **Critical Taxonomy Constraint**: Continuous run-to-failure degradation without discrete fault severity ground truth.
- **Assessment**: Unsuitable for discrete stationary 4-class classification. Labeling the onset of fault introduces subjective thresholds. Multiple bearings share a single shaft, introducing cross-bearing acoustic contamination.

### 2.4 Southeast University (SEU) Gearbox and Bearing Dataset
- **Institution**: Southeast University, Nanjing, China.
- **Availability & License**: Open-source benchmark repository hosted on GitHub/Dataverse.
- **Structure**: Drivetrain Dynamics Simulator (DDS) rig containing both planetary gear and rolling bearing defect modules.
- **Bearing Categories**:
  1. **Healthy / Normal**
  2. **Ball Fault (BF)**
  3. **Inner Race Fault (IF)**
  4. **Outer Race Fault (OF)**
  5. Combination Fault (Inner + Outer)
- **Sampling Rate**: 5,120 Hz (8 synchronized channels including drive-end acceleration).
- **Operating Conditions**: Evaluated under 20 Hz – 0 V and 30 Hz – 2 V speed-load settings.
- **Assessment**: **Directly compatible 4-class taxonomy!** Captures identical physical failure modes on a completely different mechanical testbed.

### 2.5 Jiangnan University (JNU) Bearing Dataset
- **Institution**: School of Mechanical Engineering, Jiangnan University, Wuxi, China.
- **Availability & License**: Open academic repository.
- **Structure**: Induction motor test rig with single-point wire-EDM localized defects:
  1. **Normal State (Normal)**
  2. **Ball Fault (RF - Roller Fault)**
  3. **Inner Race Fault (IF)**
  4. **Outer Race Fault (OF)**
- **Sampling Rate**: 50,000 Hz.
- **Operating Conditions**: 600 RPM, 800 RPM, and 1000 RPM.
- **Assessment**: **Directly compatible 4-class taxonomy!** High sampling rate allows decimation to InspectAI's 12 kHz standard.

---

## 3. Dataset Compatibility & Comparison Matrix

| Property | CWRU (Baseline) | Paderborn (PU) | MFPT | NASA IMS | Southeast Univ (SEU) | Jiangnan Univ (JNU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct 4-Class Compatibility** | **Yes** (Native) | No (3-Class) | No (3-Class) | No (Run-to-failure) | **Yes** | **Yes** |
| **Normal Class** | Yes | Yes (K001-K006) | Yes (3 runs) | Yes (Early life) | Yes | Yes |
| **Ball Fault Class** | Yes | **No** | **No** | Test 1 Roller | **Yes** | **Yes** |
| **Inner Race Class** | Yes | Yes (KI01-KI08) | Yes (7 runs) | Test 1 Bearing 3 | **Yes** | **Yes** |
| **Outer Race Class** | Yes | Yes (KA01-KA09) | Yes (10 runs) | Test 2, 3 | **Yes** | **Yes** |
| **Native Sampling Rate** | 12 kHz / 48 kHz | 64 kHz | 48.8 / 97.6 kHz | 20 kHz | 5.12 kHz | 50 kHz |
| **Resampling to 12 kHz** | None (Native) | Decimate ($5.33\times$) | Decimate ($4\times/8\times$) | Polyphase ($1.66\times$) | Requires Upsampling | Decimate ($4.16\times$) |
| **Defect Generation** | Spark EDM | EDM + Real Fatigue | Mechanical Notch | Natural Degradation | EDM wire-cut | EDM wire-cut |
| **Bearing Geometry** | 6205-2RS JEM SKF | 6203 | Custom rig | Rexnord ZA-2115 | ER-16K | N205 |
| **Motor Speed Range** | 1720 - 1797 RPM | 900 - 1500 RPM | ~1800 RPM | 2000 RPM | 1200 - 1800 RPM | 600 - 1000 RPM |
| **Suitability Tier** | Reference | Secondary (3-Class) | Secondary (3-Class) | Incompatible | **Primary Candidate A** | **Primary Candidate B** |

---

## 4. Recommended External Validation Strategy

### 4.1 Primary Selection: Two-Track Validation Architecture

Rather than forcing an incompatible 3-class dataset into a 4-class network, we formulate a two-track strategy:

```
                                      +-----------------------------------------+
                                      |     InspectAI External Validation       |
                                      +-----------------------------------------+
                                                           |
                      +------------------------------------+------------------------------------+
                      |                                                                         |
                      v                                                                         v
      [Track 1: Direct 4-Class Transfer]                                        [Track 2: Intra-Rig Load Generalization]
  Dataset: Jiangnan (JNU) or Southeast (SEU)                                 Dataset: Expanded CWRU Multi-Load Suite
  - 4 Classes: Normal, Ball, Inner, Outer                                    - 4 Classes across 0, 1, 2, and 3 HP
  - Evaluates cross-rig mechanical transfer                                  - Disentangles defect severity from load
  - Strict instance normalization                                            - Tests if model generalizes across loads
```

1. **Track 1 (Cross-Rig Generalization on SEU or JNU)**:
   - Evaluates genuine domain generalization on a completely different physical testbed with different bearing geometry, housing stiffness, and sensor placement.
   - Recommended dataset: **Jiangnan University (JNU)** due to its 50 kHz sampling rate (clean polyphase decimation to 12 kHz or 10 kHz without synthetic upsampling artifacts).
   - If JNU is unavailable or licensing restricts download, **Southeast University (SEU)** serves as the direct alternative.

2. **Track 2 (Rig-Controlled Defect Disentanglement on Expanded CWRU)**:
   - Before expecting cross-rig transfer, the model must prove it can classify across load levels and unseen defect diameters within the same rig.
   - Download the remaining untouched 12 kHz CWRU recordings at 1 HP, 2 HP, and 3 HP across 0.007", 0.014", and 0.021" defects.
   - Partition by motor load (e.g. train on 0 HP and 3 HP; test on 1 HP and 2 HP) to isolate load-zone domain shift from defect severity.

---

## 5. Signal Preprocessing & Normalization Adaptation Plan

In cross-dataset validation, the existing global normalizer (`inspectai_final_normalizer.json`) **must not be used**. Sensor gains, charge amplifier sensitivities, and physical units differ across testbeds.

### 5.1 Proposed Conditioning Pipeline
1. **Resampling**: Polyphase anti-aliasing resampling to match a standardized sampling frequency (e.g. $F_s = 12{,}000\text{ Hz}$).
2. **DC Removal**: Subtract window mean ($x[n] - \bar{x}$) to remove sensor offset.
3. **Bandpass Filtering**: 4th-order zero-phase Butterworth bandpass ($[20\text{ Hz}, 5{,}000\text{ Hz}]$) to remove low-frequency motor rumble and out-of-band acoustic noise.
4. **Window-Level Z-Score Normalization (Instance Normalization)**:
   $$\tilde{x}[n] = \frac{x[n] - \mu_w}{\sigma_w + \epsilon}$$
   where $\mu_w$ and $\sigma_w$ are computed *independently for each individual 2,048-sample window*.
   *Scientific Justification*: Instance normalization forces the 1D CNN to classify based on **temporal waveform geometry, pulse periodicity, and spectral distribution** rather than absolute vibrational energy (RMS). This directly eliminates the amplitude shortcut that caused the 0.014-inch test failure.

---

## 6. Leakage-Safe Experimental Protocol

Any future evaluation in Phase 10 must strictly obey these leakage-prevention guarantees:

1. **Recording-Level & Machine-Level Grouping**: Windows from the same physical run or recording must never appear in both training and testing sets.
2. **Temporal Window Isolation**: If a recording is partitioned into train/val, a minimum buffer gap equal to the window length (2,048 samples) must separate splits to prevent overlapping boundary leakage. In cross-dataset testing, the entire external dataset is holdout.
3. **No Target Leakage in Preprocessing**: Any signal statistics (mean, variance, filter coefficients) must be computed either strictly within the training set or strictly on a per-window basis (instance normalization).
4. **Frozen Holdout Guarantee**: The existing CWRU holdout test set (`97`, `169`, `185`, `197`) must remain permanently frozen with its historical 39.73% accuracy and 34.04% Macro-F1. Future generalization experiments must be reported as a separate, distinct benchmark table.

---

## 7. Resource & Feasibility Requirements

- **Compute**: All proposed preprocessing, decimation, and inference pipelines run on 100% CPU on standard hardware (Intel Core i5/i7 or AMD Ryzen, 16 GB RAM).
- **Storage**:
  - JNU Bearing Dataset: ~500 MB compressed.
  - SEU Bearing Dataset: ~800 MB compressed.
  - Expanded CWRU Multi-Load files: ~60 MB.
- **Execution Time**: Resampling and feature extraction take <60 seconds per dataset on CPU.

---

## 8. Limitations & Risks

1. **Fundamental Mechanical Transfer Barrier**: Rolling element bearings with different pitch diameters, contact angles, and ball counts have different kinematic defect characteristic equations:
   $$f_{\text{BPFO}} = \frac{N_b}{2} f_r \left(1 - \frac{d}{D}\cos\alpha\right)$$
   $$f_{\text{BPFI}} = \frac{N_b}{2} f_r \left(1 + \frac{d}{D}\cos\alpha\right)$$
   A 1D CNN trained purely on time-series vibration of an SKF 6205 bearing ($d/D \approx 0.151, N_b=9$) will encounter shifted characteristic impact frequencies on an ER-16K bearing ($N_b=9$) or an N205 bearing ($N_b=13$).
2. **Domain Adaptation Requirement**: Zero-shot direct transfer across different machine rigs is expected to fail without domain adaptation techniques (e.g. Correlation Alignment, Maximum Mean Discrepancy, or order tracking normalization).
3. **Scientific Reporting Standard**: All future evaluation reports must explicitly distinguish between **in-domain cross-validation**, **held-out defect cross-validation**, and **cross-rig transfer validation**.


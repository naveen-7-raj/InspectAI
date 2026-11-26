# InspectAI – Phase 11.1: Multi-Load Cross-Validation Design & Integrity Report

**Document ID:** `DOC-VAL-11.1`  
**Phase:** 11.1 – Multi-Load Cross-Validation Design & Integrity Audit  
**Author:** AI/ML Engineering Team  
**Status:** COMPLETE & VERIFIED  
**Date:** November 2026  
**Audience:** AI/ML Engineer Intern Recruiters & Industrial Machine Health Monitoring Specialists  

---

## Executive Summary

Phase 10.5 diagnosed that the primary bottleneck in the baseline 1D CNN's generalization was dataset confounding: in the initial 0 HP development set, fault location and fault severity were 100% collinear in cross-validation folds. Phase 10.6 addressed this by acquiring 27 authentic multi-load recordings from the CWRU Bearing Data Center, creating an expanded 40-recording catalog.

**Phase 11.1 designs and validates a leakage-safe Leave-One-Load-Out (LOLO) cross-validation protocol** using these 40 recordings. 

### Core Protocol Principles
1. **Strict Holdout Quarantine:** The 4 historical holdout test recordings (`97`, `169`, `185`, `197`) are permanently quarantined from all development folds. Their SHA-256 hashes are verified against historical inventories.
2. **True Operating Condition Generalization:** Folds are partitioned strictly by motor load (HP). In each fold, the model is trained on two or three motor loads and evaluated on an entirely unseen motor load.
3. **Exclusion of 0 HP Validation:** No 0 HP validation fold is created because the only 0 HP Normal recording (`97.mat`) is part of the protected historical holdout. Evaluating on 0 HP without holdout contamination would leave the validation set with zero Normal samples.
4. **Zero Window or Recording Leakage:** Recordings and their overlapping windows are strictly partitioned. No window crosses partition boundaries.
5. **Isolated Preprocessing Statistics:** Preprocessing normalizer parameters ($\mu, \sigma$) are computed strictly on the training partitions of each fold.
6. **Zero Model Training:** In accordance with Phase 11.1 scope boundaries, no neural networks were trained or evaluated.

---

## 1. Dataset Partitioning Architecture

The 40 authentic CWRU recordings are partitioned into two tiers:
- **Historical Holdout Test Set (4 recordings):** Quarantined exclusively for post-experimental independent auditing.
- **Multi-Load Development Pool (36 recordings):** Used to construct the 3 Leave-One-Load-Out (LOLO) folds.

```
Expanded CWRU Catalog (40 Recordings, 133.82 MB)
├── Protected Historical Holdout [TEST ONLY] (4 recordings, 818 windows)
│   ├── Normal (0 HP): 97.mat
│   ├── Inner Race 0.014" (0 HP): 169.mat
│   ├── Ball Fault 0.014" (0 HP): 185.mat
│   └── Outer Race 0.014" (0 HP): 197.mat
│
└── Multi-Load Development Pool (36 recordings, 8,448 windows)
    ├── Load 0 HP: 6 recordings (1,410 windows) [Faults only: 0.007" & 0.021"]
    ├── Load 1 HP: 10 recordings (2,345 windows) [1 Normal, 9 Faults: 0.007", 0.014", 0.021"]
    ├── Load 2 HP: 10 recordings (2,343 windows) [1 Normal, 9 Faults: 0.007", 0.014", 0.021"]
    └── Load 3 HP: 10 recordings (2,350 windows) [1 Normal, 9 Faults: 0.007", 0.014", 0.021"]
```

---

## 2. Leave-One-Load-Out (LOLO) Fold Specifications

Three distinct validation folds are defined, each holding out one operational motor load:

| Parameter | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| **Validation Motor Load** | **1 HP** (1772 RPM) | **2 HP** (1750 RPM) | **3 HP** (1725 RPM) |
| **Training Motor Loads** | **0 HP, 2 HP, 3 HP** | **0 HP, 1 HP, 3 HP** | **0 HP, 1 HP, 2 HP** |
| **Validation Recordings** | 10 recordings | 10 recordings | 10 recordings |
| **Training Recordings** | 26 recordings | 26 recordings | 26 recordings |
| **Validation Windows** | 2,345 windows | 2,343 windows | 2,350 windows |
| **Training Windows** | 6,103 windows | 6,105 windows | 6,098 windows |
| **Total Dev Windows** | 8,448 windows | 8,448 windows | 8,448 windows |
| **Train/Val Disjointness** | Verified (0 overlap) | Verified (0 overlap) | Verified (0 overlap) |
| **Holdout Contamination** | Verified (0 holdouts) | Verified (0 holdouts) | Verified (0 holdouts) |

### 2.1 Recording-Level Split Inventory

#### Fold 1 (Validation Load: 1 HP)
- **Validation Recordings (10):**
  - *Normal (1):* `98`
  - *Inner Race (3):* `106` (0.007"), `170` (0.014"), `210` (0.021")
  - *Ball Fault (3):* `119` (0.007"), `186` (0.014"), `223` (0.021")
  - *Outer Race (3):* `131` (0.007"), `198` (0.014"), `235` (0.021")
- **Training Recordings (26):**
  - *Normal (2):* `99` (2 HP), `100` (3 HP)
  - *Inner Race (8):* `105`, `209` (0 HP); `107`, `171`, `211` (2 HP); `108`, `172`, `212` (3 HP)
  - *Ball Fault (8):* `118`, `222` (0 HP); `120`, `187`, `224` (2 HP); `121`, `188`, `225` (3 HP)
  - *Outer Race (8):* `130`, `234` (0 HP); `132`, `199`, `236` (2 HP); `133`, `200`, `237` (3 HP)

#### Fold 2 (Validation Load: 2 HP)
- **Validation Recordings (10):**
  - *Normal (1):* `99`
  - *Inner Race (3):* `107` (0.007"), `171` (0.014"), `211` (0.021")
  - *Ball Fault (3):* `120` (0.007"), `187` (0.014"), `224` (0.021")
  - *Outer Race (3):* `132` (0.007"), `199` (0.014"), `236` (0.021")
- **Training Recordings (26):**
  - *Normal (2):* `98` (1 HP), `100` (3 HP)
  - *Inner Race (8):* `105`, `209` (0 HP); `106`, `170`, `210` (1 HP); `108`, `172`, `212` (3 HP)
  - *Ball Fault (8):* `118`, `222` (0 HP); `119`, `186`, `223` (1 HP); `121`, `188`, `225` (3 HP)
  - *Outer Race (8):* `130`, `234` (0 HP); `131`, `198`, `235` (1 HP); `133`, `200`, `237` (3 HP)

#### Fold 3 (Validation Load: 3 HP)
- **Validation Recordings (10):**
  - *Normal (1):* `100`
  - *Inner Race (3):* `108` (0.007"), `172` (0.014"), `212` (0.021")
  - *Ball Fault (3):* `121` (0.007"), `188` (0.014"), `225` (0.021")
  - *Outer Race (3):* `133` (0.007"), `200` (0.014"), `237` (0.021")
- **Training Recordings (26):**
  - *Normal (2):* `98` (1 HP), `99` (2 HP)
  - *Inner Race (8):* `105`, `209` (0 HP); `106`, `170`, `210` (1 HP); `107`, `171`, `211` (2 HP)
  - *Ball Fault (8):* `118`, `222` (0 HP); `119`, `186`, `223` (1 HP); `120`, `187`, `224` (2 HP)
  - *Outer Race (8):* `130`, `234` (0 HP); `131`, `198`, `235` (1 HP); `132`, `199`, `236` (2 HP)

---

## 3. Class and Window Balance Analysis

Every fold maintains representation across all 4 machine health conditions in both training and validation partitions:

### 3.1 Fold 1 Windows
- **Validation (1 HP):** 
  - Normal: 233 windows (9.9%)
  - Inner Race: 703 windows (30.0%)
  - Ball Fault: 703 windows (30.0%)
  - Outer Race: 706 windows (30.1%)
  - *Total Validation Windows:* **2,345**
- **Training (0, 2, 3 HP):**
  - Normal: 467 windows (7.7%)
  - Inner Race: 1,877 windows (30.8%)
  - Ball Fault: 1,879 windows (30.8%)
  - Outer Race: 1,880 windows (30.8%)
  - *Total Training Windows:* **6,103**

### 3.2 Fold 2 Windows
- **Validation (2 HP):**
  - Normal: 233 windows (9.9%)
  - Inner Race: 703 windows (30.0%)
  - Ball Fault: 704 windows (30.0%)
  - Outer Race: 703 windows (30.0%)
  - *Total Validation Windows:* **2,343**
- **Training (0, 1, 3 HP):**
  - Normal: 467 windows (7.7%)
  - Inner Race: 1,877 windows (30.7%)
  - Ball Fault: 1,878 windows (30.8%)
  - Outer Race: 1,883 windows (30.8%)
  - *Total Training Windows:* **6,105**

### 3.3 Fold 3 Windows
- **Validation (3 HP):**
  - Normal: 234 windows (10.0%)
  - Inner Race: 706 windows (30.0%)
  - Ball Fault: 704 windows (30.0%)
  - Outer Race: 706 windows (30.0%)
  - *Total Validation Windows:* **2,350**
- **Training (0, 1, 2 HP):**
  - Normal: 466 windows (7.6%)
  - Inner Race: 1,874 windows (30.7%)
  - Ball Fault: 1,878 windows (30.8%)
  - Outer Race: 1,880 windows (30.8%)
  - *Total Training Windows:* **6,098**

---

## 4. Defect Severity & Imbalance Audit

To avoid repeating the methodological error of Phase 10.3/10.4 (assuming multi-load data automatically removes confounding), the distribution of fault severities was audited:

### 4.1 Severity Matrix across Training and Validation

| Defect Diameter | Validation (Per Fold) | Training (Per Fold) | Root Cause / Note |
|---|---|---|---|
| **0.000" (Normal)** | 1 recording (~233 windows) | 2 recordings (~467 windows) | Normal baseline; only 1 recording exists per load |
| **0.007" (Early Stage)** | 3 recordings (~704 windows) | 9 recordings (~2,112 windows) | 1 per fault class at each load |
| **0.014" (Intermediate)** | 3 recordings (~703 windows) | 6 recordings (~1,407 windows) | **Imbalance:** 0 HP intermediate faults are quarantined in holdout (`169`, `185`, `197`) |
| **0.021" (Severe Fault)** | 3 recordings (~704 windows) | 9 recordings (~2,112 windows) | 1 per fault class at each load |

### 4.2 Identified Limitations & Engineering Findings

1. **Intermediate Fault (0.014") Representation in Training:**
   Because recordings `169.mat`, `185.mat`, and `197.mat` (the 0 HP 0.014" fault recordings) are permanently quarantined in the historical holdout, the 0 HP load level in the development pool contains only 0.007" and 0.021" faults. Consequently, training sets across all three folds possess 9 recordings each of 0.007" and 0.021" faults, but only 6 recordings of 0.014" faults.
2. **Normal Class Asymmetry:**
   The CWRU benchmark contains only one baseline Normal recording per motor load level (`97` at 0 HP, `98` at 1 HP, `99` at 2 HP, `100` at 3 HP). With `97` quarantined in the holdout, development folds have exactly 1 Normal recording in validation and 2 in training (~7.7% of total windows). Class-weighted cross-entropy loss remains mandatory to prevent the network from favoring the majority fault classes.
3. **Impossibility of 0 HP Validation:**
   A 0 HP validation fold cannot be constructed without contaminating the test holdout. Creating a 0 HP validation fold without `97` would result in a validation set entirely devoid of Normal samples.
4. **Collinearity Warning:**
   While multi-load cross-validation decouples load from severity within each fold, empirical validation in Phase 11.2 must measure whether the CNN learns invariant bearing kinematic signatures rather than load-dependent resonance artifacts.

---

## 5. Preprocessing & Normalization Integrity

The multi-load dataset pipeline follows strict digital signal processing rules:

1. **Anti-Aliasing & Sampling Rate Harmonization:**
   Raw CWRU Normal recordings are sampled at 48,000 Hz, while Drive-End fault recordings are sampled at 12,000 Hz. `SignalCleaner` applies polyphase downsampling (`resample_poly` with factor $1/4$) accompanied by an internal Chebyshev anti-aliasing filter before signal windowing.
2. **Bandpass Filtering:**
   A 4th-order Butterworth bandpass filter (20 Hz – 5,000 Hz) eliminates motor shaft DC bias and high-frequency transducer noise above the Nyquist limit.
3. **Window Extraction:**
   Fixed windows of length $L = 2048$ samples ($170.67$ ms) are sliced with a stride of $S = 512$ samples (75% overlap). No partial or zero-padded windows are permitted.
4. **Leakage-Proof Normalizer Calibration:**
   The `VibrationNormalizer` computes global scalar statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) strictly from training window arrays. Validation signals are transformed using the frozen training parameters:
   $$x_{\text{val, norm}} = \frac{x_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$

---

## 6. Deliverables and Artifact Inventory

All Phase 11.1 deliverables have been generated, tested, and stored in the project workspace:

1. **Splitter Module:** [src/training/multiload_splitter.py](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/src/training/multiload_splitter.py)
2. **Verification CLI:** [scripts/verify_multiload_splits.py](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/scripts/verify_multiload_splits.py)
3. **Splits Configuration JSON:** [outputs/experiments/phase11_1/multiload_splits.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_1/multiload_splits.json)
4. **Fold Summary CSV:** [outputs/experiments/phase11_1/lolo_fold_summary.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_1/lolo_fold_summary.csv)
5. **Recording Distribution CSV:** [outputs/experiments/phase11_1/lolo_recording_distribution.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_1/lolo_recording_distribution.csv)
6. **Defect Severity Analysis JSON:** [outputs/experiments/phase11_1/lolo_defect_severity_analysis.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_1/lolo_defect_severity_analysis.json)
7. **Verification Audit Report JSON:** [outputs/experiments/phase11_1/split_verification_report.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_1/split_verification_report.json)
8. **Unit Test Suite:** [tests/test_multiload_splitter.py](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/tests/test_multiload_splitter.py) (8/8 tests passing).

---

## 7. Next Steps: Transition to Phase 11.2

With the Leave-One-Load-Out validation protocol rigorously designed and verified against data leakage, the next phase can proceed:
- **Phase 11.2:** Multi-Load 1D CNN Baseline Training & Cross-Load Generalization Evaluation across Folds 1, 2, and 3.


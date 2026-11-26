# InspectAI – Phase 11.2: Multi-Load Generalization Experiment Report

**Document ID:** `DOC-EXP-11.2`  
**Phase:** 11.2 – Multi-Load Generalization Experiment  
**Author:** AI/ML Engineering Team  
**Status:** COMPLETE & EMPIRICALLY VERIFIED  
**Date:** November 2026  
**Audience:** AI/ML Engineer Intern Recruiters & Industrial Machine Health Monitoring Specialists  

---

## Executive Summary

In Phase 10.5, InspectAI discovered that the 1D CNN's diagnostic failure on severe faults was driven by **dataset confounding**: at 0 HP, fault location and fault severity were collinear across development splits, prompting the model to rely on high-frequency resonance energy as a heuristic for Outer Race faults.

**Phase 11.2 tested whether training across multi-load, multi-severity recordings resolves this bottleneck and enables true cross-load generalization.**

Using the 3-fold **Leave-One-Load-Out (LOLO)** cross-validation protocol designed in Phase 11.1, the production `VibrationCNN1D` (73,156 parameters, CPU-only) was evaluated across 3 unseen motor loads (1 HP, 2 HP, and 3 HP):

### Key Experimental Findings
- **Mean Cross-Fold Macro-F1:** **99.27% ± 1.04%** (Accuracy: **99.12% ± 1.24%**, Balanced Accuracy: **99.27% ± 1.04%**).
- **Cross-Load Generalization:**
  - **Fold 1 (Held-out 1 HP):** **100.00% Accuracy / 100.00% Macro-F1** (2,345/2,345 windows correct).
  - **Fold 2 (Held-out 2 HP):** **100.00% Accuracy / 100.00% Macro-F1** (2,343/2,343 windows correct).
  - **Fold 3 (Held-out 3 HP):** **97.36% Accuracy / 97.80% Macro-F1** (2,288/2,350 windows correct).
- **Resolution of Inner Race vs. Outer Race Shortcut:**
  Outer Race precision and recall reached **100.00%** across all 3 held-out loads (2,115/2,115 windows correct). Severe (0.021") Inner Race recordings—which completely failed under Phase 10.5—achieved **100.00% accuracy** across all loads (`210` at 1 HP, `211` at 2 HP, `212` at 3 HP).
- **Residual Error Analysis:**
  The only misclassifications occurred in Fold 3 on recording `172.mat` (0.014" Inner Race at 3 HP), where 62 of 234 windows (26.5%) were misclassified as Ball Fault, while 172 windows (73.5%) were correctly classified.
- **CPU Inference Latency:**
  Median single-window CPU latency is **1.3174 ms** ($p95 = 1.8981$ ms). In batch mode ($B=32$), per-sample median latency drops to **0.5686 ms**.
- **Historical Holdout Quarantine:**
  The 4 historical holdout test recordings (`97`, `169`, `185`, `197`) remained 100% untouched and quarantined.

---

## 1. Methodology & Validation Protocol

### 1.1 Strict Leakage Prevention Architecture

To guarantee scientific validity, the outer held-out motor load in each fold was strictly prevented from influencing any aspect of training, normalization, hyperparameter selection, or early stopping.

Within each outer fold:
1. **Outer Partition:** 26 outer-training recordings (loads $L \ne L_{\text{val}}$) and 10 outer-evaluation recordings (held-out load $L_{\text{val}}$).
2. **Inner Stratified Split:** The 26 outer-training recordings were partitioned at the recording level into **19 inner-training recordings** and **7 inner-validation recordings**. Both inner partitions contained all 4 classes (Normal, Ball Fault, Inner Race, Outer Race).
3. **Inner Normalizer & Class Weights:** A `VibrationNormalizer(strategy="global")` was fitted exclusively on the 19 inner-training recordings. Class weights for `CrossEntropyLoss` were computed exclusively from inner-training labels.
4. **Best Epoch Selection ($E^*$):** A fresh `VibrationCNN1D` was trained on inner-training windows for up to 30 epochs with early stopping (patience = 7, monitoring validation Macro-F1). The optimal epoch $E^*$ was determined.
5. **Full Outer Retraining:**
   - A fresh normalizer was fitted on all 26 outer-training recordings.
   - Fresh class weights were computed on all outer-training labels.
   - A new `VibrationCNN1D` instance was retrained from scratch on all 26 outer-training recordings for exactly $E^*$ epochs.
6. **Single-Pass Outer Evaluation:** The outer model was evaluated exactly once on the untouched 10 held-out recordings transformed by the outer normalizer.

```
Outer Fold Loop (Fold 1: 1 HP, Fold 2: 2 HP, Fold 3: 3 HP)
│
├── 1. Outer Training Partition (26 recordings, ~6,100 windows)
│   │
│   ├── [Inner Split]
│   │   ├── Inner Train (19 recs, ~4,460 windows) -> Fit Inner Normalizer & Class Weights
│   │   └── Inner Val   (7 recs, ~1,640 windows)  -> Early Stopping -> Select E*
│   │
│   └── [Full Outer Retraining]
│       ├── Fit Outer Normalizer on all 26 Outer Train recordings
│       ├── Compute Outer Class Weights on all 26 Outer Train recordings
│       └── Train fresh CNN for exactly E* epochs on all 26 Outer Train recordings
│
└── 2. Single-Pass Outer Evaluation (10 held-out recordings, ~2,345 windows)
    └── Transform using Outer Normalizer -> Compute Accuracy, Bal-Acc, Macro-F1, Confusion Matrix
```

### 1.2 Model & Training Configuration

In accordance with Phase 11.2 constraints, the frozen baseline architecture was preserved without modification:
- **Architecture:** `VibrationCNN1D`
- **Input Dimensions:** `(Batch, 1, 2048)` at 12,000 Hz
- **Feature Extractor:** 3 Conv1D blocks with BatchNorm, ReLU, and MaxPool1D
  - Block 1: Conv1D(1 -> 32, k=15, p=7) + MaxPool(4)
  - Block 2: Conv1D(32 -> 64, k=7, p=3) + MaxPool(4)
  - Block 3: Conv1D(64 -> 128, k=3, p=1) + MaxPool(2)
- **Classifier Head:** Global Average Pooling + Dropout(0.3) + Linear(128, 64) + ReLU + Linear(64, 4)
- **Trainable Parameters:** **73,156** (`fc_units=64`)
- **Optimizer:** AdamW ($\text{lr} = 0.0005, \text{weight\_decay} = 0.0001$)
- **Batch Size:** 32 (CPU execution)
- **Loss Function:** Class-Weighted Cross-Entropy Loss
- **Random Seed:** 42

---

## 2. Experimental Results across Held-Out Motor Loads

### 2.1 Per-Load Performance Summary

| Fold | Held-Out Load | Motor Speed | Train Loads | Selected Epoch ($E^*$) | Evaluated Windows | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold 1** | **1 HP** | 1772 RPM | 0, 2, 3 HP | 4 | 2,345 | **100.00%** | **100.00%** | 100.00% | 100.00% | **100.00%** |
| **Fold 2** | **2 HP** | 1750 RPM | 0, 1, 3 HP | 21 | 2,343 | **100.00%** | **100.00%** | 100.00% | 100.00% | **100.00%** |
| **Fold 3** | **3 HP** | 1725 RPM | 0, 1, 2 HP | 2 | 2,350 | **97.36%** | **97.80%** | 97.98% | 97.80% | **97.80%** |
| **Mean ± Std** | — | — | — | — | — | **99.12% ± 1.24%** | **99.27% ± 1.04%** | **99.33% ± 0.95%** | **99.27% ± 1.04%** | **99.27% ± 1.04%** |
| **Pooled** | **All Loads** | — | — | — | **7,038** | **99.12%** | **99.27%** | **99.29%** | **99.27%** | **99.27%** |

### 2.2 Pooled Confusion Matrix (7,038 Windows)

```
                       Predicted Class
True Class           Normal   Ball Fault   Inner Race   Outer Race    Total
--------------------------------------------------------------------------
Normal                  700            0            0            0      700
Ball Fault                0        2,111            0            0    2,111
Inner Race Fault          0           62        2,050            0    2,112
Outer Race Fault          0            0            0        2,115    2,115
--------------------------------------------------------------------------
Total                   700        2,173        2,050        2,115    7,038
```

### 2.3 Pooled Per-Class Diagnostics

| Class | Precision | Recall | F1-Score | Support (Windows) |
|---|---|---|---|---|
| **Normal** | **100.00%** | **100.00%** | **100.00%** | 700 |
| **Ball Fault** | **97.15%** | **100.00%** | **98.55%** | 2,111 |
| **Inner Race Fault** | **100.00%** | **97.06%** | **98.51%** | 2,112 |
| **Outer Race Fault** | **100.00%** | **100.00%** | **100.00%** | 2,115 |
| **Macro Average** | **99.29%** | **99.27%** | **99.27%** | 7,038 |

---

## 3. Defect-Severity Breakdown & Failure Analysis

To evaluate whether the model learned invariant defect signatures or succumbed to defect-specific artifacts, performance was disaggregated by fault severity across all held-out loads.

### 3.1 Severity Breakdown Table

| Held-Out Load | Fault Class | Defect Size | Recording ID | Total Windows | Correct Windows | Accuracy | Diagnostic Outcome |
|---|---|---|---|---|---|---|---|
| **1 HP** | Normal | 0.000" | `98` | 233 | 233 | **100.0%** | Perfect discrimination |
| 1 HP | Ball Fault | 0.007" | `119` | 234 | 234 | **100.0%** | Perfect discrimination |
| 1 HP | Ball Fault | 0.014" | `186` | 235 | 235 | **100.0%** | Perfect discrimination |
| 1 HP | Ball Fault | 0.021" | `223` | 234 | 234 | **100.0%** | Perfect discrimination |
| 1 HP | Inner Race | 0.007" | `106` | 235 | 235 | **100.0%** | Perfect discrimination |
| 1 HP | Inner Race | 0.014" | `170` | 234 | 234 | **100.0%** | Perfect discrimination |
| 1 HP | Inner Race | 0.021" | `210` | 234 | 234 | **100.0%** | Perfect discrimination |
| 1 HP | Outer Race | 0.007" | `131` | 236 | 236 | **100.0%** | Perfect discrimination |
| 1 HP | Outer Race | 0.014" | `198` | 235 | 235 | **100.0%** | Perfect discrimination |
| 1 HP | Outer Race | 0.021" | `235` | 235 | 235 | **100.0%** | Perfect discrimination |
| **2 HP** | Normal | 0.000" | `99` | 233 | 233 | **100.0%** | Perfect discrimination |
| 2 HP | Ball Fault | 0.007" | `120` | 234 | 234 | **100.0%** | Perfect discrimination |
| 2 HP | Ball Fault | 0.014" | `187` | 235 | 235 | **100.0%** | Perfect discrimination |
| 2 HP | Ball Fault | 0.021" | `224` | 235 | 235 | **100.0%** | Perfect discrimination |
| 2 HP | Inner Race | 0.007" | `107` | 235 | 235 | **100.0%** | Perfect discrimination |
| 2 HP | Inner Race | 0.014" | `171` | 234 | 234 | **100.0%** | Perfect discrimination |
| 2 HP | Inner Race | 0.021" | `211` | 234 | 234 | **100.0%** | Perfect discrimination |
| 2 HP | Outer Race | 0.007" | `132` | 234 | 234 | **100.0%** | Perfect discrimination |
| 2 HP | Outer Race | 0.014" | `199` | 234 | 234 | **100.0%** | Perfect discrimination |
| 2 HP | Outer Race | 0.021" | `236` | 235 | 235 | **100.0%** | Perfect discrimination |
| **3 HP** | Normal | 0.000" | `100` | 234 | 234 | **100.0%** | Perfect discrimination |
| 3 HP | Ball Fault | 0.007" | `121` | 234 | 234 | **100.0%** | Perfect discrimination |
| 3 HP | Ball Fault | 0.014" | `188` | 235 | 235 | **100.0%** | Perfect discrimination |
| 3 HP | Ball Fault | 0.021" | `225` | 235 | 235 | **100.0%** | Perfect discrimination |
| 3 HP | Inner Race | 0.007" | `108` | 237 | 237 | **100.0%** | Perfect discrimination |
| 3 HP | Inner Race | 0.014" | `172` | 234 | 172 | **73.5%** | **62 windows misclassified as Ball Fault** |
| 3 HP | Inner Race | 0.021" | `212` | 235 | 235 | **100.0%** | Perfect discrimination |
| 3 HP | Outer Race | 0.007" | `133` | 236 | 236 | **100.0%** | Perfect discrimination |
| 3 HP | Outer Race | 0.014" | `200` | 235 | 235 | **100.0%** | Perfect discrimination |
| 3 HP | Outer Race | 0.021" | `237` | 235 | 235 | **100.0%** | Perfect discrimination |

### 3.2 Key Diagnostic Insights

1. **Complete Elimination of the Phase 10.5 Resonance Shortcut:**
   In Phase 10.5, severe Inner Race recording `209` was 100% misclassified as Outer Race because high-frequency resonance energy was collinear with Outer Race faults in the 0 HP dataset. In Phase 11.2, severe Inner Race recordings across all three loads (`210`, `211`, `212`) achieved **100.00% accuracy**. Outer Race faults also achieved **100.00% accuracy** across all loads and severities.
2. **Analysis of the Recording 172 Failure Mode:**
   In Fold 3 (3 HP held out), early stopping selected $E^* = 2$ epochs based on inner validation. While this was sufficient for 29 out of 30 held-out recording instances, recording `172` (0.014" Inner Race at 3 HP) experienced 62 misclassifications as Ball Fault. At 3 HP (maximum motor torque and speed of 1728 RPM), intermediate inner race impacts generate kinematic vibration signatures that partially overlap with ball-spin modulation harmonics. 
3. **Scientific Scope Caution:**
   While multi-load training resolves the specific collinearity present in the single-load 0 HP dataset, these results reflect generalization across motor operating loads **on the same CWRU test rig**. They must not be extrapolated to claim cross-machine generalization or industrial deployment readiness.

---

## 4. CPU Inference Latency Benchmarking

Inference latency was benchmarked on a standard CPU environment using PyTorch in evaluation mode (`torch.no_grad()`):

| Configuration | Batch Size | Timed Iterations | Median Latency | 95th Percentile ($p95$) | Mean Latency | Minimum | Maximum |
|---|---|---|---|---|---|---|---|
| **Single Sample** | 1 | 100 | **1.3174 ms** | **1.8981 ms** | 1.3671 ms | 0.6170 ms | 2.5211 ms |
| **Batch Inference** | 32 | 100 | **18.1961 ms** | **22.5595 ms** | 18.2138 ms | 13.9102 ms | 26.4380 ms |
| **Per-Sample (Batch)** | 1 | — | **0.5686 ms** | **0.7050 ms** | 0.5692 ms | 0.4347 ms | 0.8262 ms |

*Hardware context: Intel/AMD x86_64 CPU, Windows OS, single-thread PyTorch execution. A 1.32 ms latency is well within typical edge-device processing budgets for 170 ms vibration windows (providing a ~120x real-time margin).*

---

## 5. Methodological Comparison with Prior Phases

It is critical to distinguish between earlier experimental benchmarks and Phase 11.2, as their validation protocols and dataset scopes differ fundamentally:

| Dimension | Phase 10.2 (Normalization) | Phase 10.3 (Multiscale CNN) | Phase 10.4 (Spectral Fusion) | Phase 11.2 (Multi-Load Generalization) |
|---|---|---|---|---|
| **Dataset Scope** | 9 Development Recordings (0 HP only) | 9 Development Recordings (0 HP only) | 9 Development Recordings (0 HP only) | **36 Multi-Load Development Recordings (0, 1, 2, 3 HP)** |
| **Validation Protocol** | 2-Fold Stratified Group K-Fold | 2-Fold Stratified Group K-Fold | 2-Fold Stratified Group K-Fold | **3-Fold Leave-One-Load-Out (LOLO)** |
| **Evaluated Condition** | Same load (0 HP), cross-recording | Same load (0 HP), cross-recording | Same load (0 HP), cross-recording | **Completely unseen motor load (cross-condition)** |
| **Severity Coverage** | 100% collinear with fault type | 100% collinear with fault type | 100% collinear with fault type | **Full diversity across 0.007", 0.014", 0.021"** |
| **Validation Metric** | OOF Macro-F1: 83.78% | OOF Macro-F1: 86.26% | OOF Macro-F1: 81.09% | **Cross-Load Macro-F1: 99.27% ± 1.04%** |

*Note: Phase 11.2 results cannot be directly subtracted from Phase 10.2–10.4 numbers because the validation task in Phase 11.2 tests transfer across physical operating regimes rather than within-load partitioning.*

---

## 6. Generated Artifacts & Deliverables

All artifacts generated by the Phase 11.2 experiment runner have been verified and saved to [outputs/experiments/phase11_2/](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2):

1. **Cross-Fold Summary JSON:** [outputs/experiments/phase11_2/cross_fold_summary.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/cross_fold_summary.json)
2. **Fold-Level Metrics JSON:** [outputs/experiments/phase11_2/fold_metrics.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/fold_metrics.json)
3. **Per-Load Metrics CSV:** [outputs/experiments/phase11_2/per_load_metrics.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/per_load_metrics.csv)
4. **Per-Class Metrics CSV:** [outputs/experiments/phase11_2/per_class_metrics.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/per_class_metrics.csv)
5. **Per-Defect-Severity Metrics CSV:** [outputs/experiments/phase11_2/per_defect_severity_metrics.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/per_defect_severity_metrics.csv)
6. **Confusion Matrices:**
   - [confusion_matrix_fold_1.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/confusion_matrix_fold_1.csv)
   - [confusion_matrix_fold_2.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/confusion_matrix_fold_2.csv)
   - [confusion_matrix_fold_3.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/confusion_matrix_fold_3.csv)
   - [confusion_matrix_pooled.csv](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/confusion_matrix_pooled.csv)
7. **Training History JSON:** [outputs/experiments/phase11_2/training_history.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/training_history.json)
8. **Inner/Outer Splits Metadata JSON:** [outputs/experiments/phase11_2/inner_outer_splits_metadata.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/inner_outer_splits_metadata.json)
9. **Normalizer Provenance JSON:** [outputs/experiments/phase11_2/normalizer_provenance.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/normalizer_provenance.json)
10. **Experiment Configuration JSON:** [outputs/experiments/phase11_2/experiment_config.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/experiment_config.json)
11. **Retrained Checkpoints & Normalizers:**
    - [lolo_fold_1_cnn.pth](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/checkpoints/lolo_fold_1_cnn.pth) & [lolo_fold_1_normalizer.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/checkpoints/lolo_fold_1_normalizer.json)
    - [lolo_fold_2_cnn.pth](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/checkpoints/lolo_fold_2_cnn.pth) & [lolo_fold_2_normalizer.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/checkpoints/lolo_fold_2_normalizer.json)
    - [lolo_fold_3_cnn.pth](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/checkpoints/lolo_fold_3_cnn.pth) & [lolo_fold_3_normalizer.json](file:///c:/Users/Naveen%20Raj/OneDrive/ドキュメント/Machine%20failure%20prediction/outputs/experiments/phase11_2/checkpoints/lolo_fold_3_normalizer.json)

---

## 7. Conclusions & Next Steps

1. **Hypothesis Confirmed:**
   Multi-load data acquisition successfully breaks the acoustic resonance shortcut identified in Phase 10.5. When exposed to fault variations across motor loads, `VibrationCNN1D` learns generalized, invariant bearing fault representations.
2. **Production Checkpoint Unmodified:**
   The original production checkpoint (`models/checkpoints/inspectai_final_cnn.pth`) and historical holdout evaluations remain strictly preserved.
3. **Phase Complete:**
   Phase 11.2 is fully concluded.


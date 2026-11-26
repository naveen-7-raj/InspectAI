# InspectAI – Phase 12: Release Integrity Audit Report

**Date**: November 2026  
**Auditor**: InspectAI Release Engineering  
**Scope**: Verification of dataset manifests, preprocessing pipelines, model artifacts, checksums, and benchmark reporting prior to portfolio release.

---

## 1. Executive Summary

As part of Phase 12 Release Engineering, a comprehensive audit of all project assets, code implementations, model weights, and benchmark records was conducted. The goal was to eliminate all discrepancies between historical Markdown reports and actual programmatic artifacts, ensuring the InspectAI portfolio launch is fully truthful, scientifically rigorous, and reproducible.

Seven critical areas were audited and reconciled:
1. **Historical Holdout Provenance**: Corrected descriptions regarding recordings `97`, `169`, `185`, and `197`.
2. **Development Recording Distribution**: Reconciled the 36-recording development partition against the 40-recording CWRU manifest.
3. **Preprocessing & Segmentation Parameters**: Reconciled the 75% overlap ($S=512$) vs. erroneous 50% documentation.
4. **Automated Test Count Reconciliation**: Verified the exact 9 test functions in `test_final_model.py` and 224 total tests.
5. **Training Window Yield & Class Balance**: Reconciled the 8,448 window distribution and loss-weighting mechanism.
6. **Artifact Checksums & Provenance**: Verified cryptographic SHA256 hashes and normalizer parameters on disk.
7. **Performance Claims vs. Canonical JSON Benchmarks**: Reconciled cross-fold metrics between documentation and `cross_fold_comparison.json`.

---

## 2. Detailed Audit Findings & Reconciliations

### 2.1 Historical Holdout Provenance (Recordings 97, 169, 185, 197)
- **Previous Erroneous Description**: In Phase 11 documentation, recordings `97`, `169`, `185`, and `197` were occasionally described as an "untouched" or "blind" benchmark.
- **Audit Fact**: In Phase 4.4, these exact four recordings were evaluated on the baseline `VibrationCNN1D` model (revealing 39.73% holdout accuracy and 0% recall on 0.014" race faults `169` and `197`).
- **Reconciliation**: These recordings are **quarantined historical benchmark controls**, permanently excluded from Phase 10 and Phase 11 multi-load development folds to prevent training leakage. However, they must not be advertised as a naive blind test set since their failure signatures were known and analyzed in Phase 4.4.

### 2.2 Development Recording Distribution (36 Recordings)
- **Previous Erroneous Description**: Summaries loosely asserted "9 recordings per class" in the development partition.
- **Audit Fact**: CWRU provides only four Drive-End normal baseline recordings across the entire 12k/48k collection (`97` at 0 HP, `98` at 1 HP, `99` at 2 HP, and `100` at 3 HP). With `97` quarantined, only **3 normal recordings** exist in the development partition.
- **Verified Distribution in Development Set (36 Recordings)**:
  - **Normal Baseline**: 3 recordings (`98`, `99`, `100` across 1, 2, 3 HP)
  - **Inner Race Fault**: 11 recordings (`105`, `106`, `107`, `108`, `170`, `171`, `172`, `209`, `210`, `211`, `212`)
  - **Ball Fault**: 11 recordings (`118`, `119`, `120`, `121`, `186`, `187`, `188`, `222`, `223`, `224`, `225`)
  - **Outer Race Fault**: 11 recordings (`130`, `131`, `132`, `133`, `198`, `199`, `200`, `234`, `235`, `236`, `237`)
  - Total: $3 + 11 + 11 + 11 = 36$ recordings.

### 2.3 Window Size and Overlap Reconciliation
- **Previous Erroneous Description**: Phase 11.5 prose mentioned a step size of 1,024 samples (50% overlap).
- **Audit Fact**: Programmatic inspection of `scripts/train_final_model.py` and `models/final/inspectai_final_config.json`:
  ```python
  WINDOW_SIZE = 2048
  STEP_SIZE = 512
  ```
- **Reconciliation**: The sliding window hop size is **512 samples** ($\approx 42.67\text{ ms}$ at 12 kHz), which represents **75% overlap** ($\frac{2048 - 512}{2048} = 0.75$). This matches the segmentation configuration utilized throughout InspectAI.

### 2.4 Training Window Count & Class Balancing
- **Previous Erroneous Description**: Prose claimed "2,112 windows per class, perfectly balanced".
- **Audit Fact**: Window extraction across the 36 development recordings produces exactly **8,448 windows**:
  - **Normal**: 700 windows (8.29%)
  - **Ball Fault**: 2,582 windows (30.56%)
  - **Inner Race Fault**: 2,580 windows (30.54%)
  - **Outer Race Fault**: 2,586 windows (30.61%)
  - Total: 8,448 windows.
- **Reconciliation**: Class balance was not achieved by arbitrary window subsampling or discarding fault windows. Instead, class balance was enforced during optimization via Scikit-learn balanced class weighting in `nn.CrossEntropyLoss`:
  $$\text{weight}_c = \frac{N_{\text{total}}}{C \cdot N_c} \implies [3.0171, 0.8180, 0.8186, 0.8167]$$

### 2.5 Verification of Model Parameter Count & Checksums
- **Architecture**: `SpectralFusionCNN`
- **Parameter Count**: Programmatically verified at exactly **89,604 parameters**.
- **Cryptographic SHA256 Checksums (On Disk)**:
  - `models/final/inspectai_final_model.pth`: `a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e`
  - `models/final/inspectai_final_normalizer.json`: `2b3db205cd2d95ead19c8ec5d14e4b2d9a8e6133ae0f0aef05e7c2bf0f1db3c1`
  - `models/final/inspectai_final_spectral_normalizer.json`: `ed39cd3615212d83c24e66a49c46fc0c07e4e9f33320e95a3cc0d3a6567a3fac`
  - `models/final/inspectai_final_config.json`: `b12f24a7837b768b2072edfc0c9ecd5b948d478b9b039e0d6640f0a43227a3a7`
  - `models/final/inspectai_model_card.json`: `139e6168e3bd176eeaabc83bc8f42e27c893ae363a8f60e945784f6b028367dc`
  - `models/checkpoints/inspectai_final_cnn.pth`: `6610b0d601fcfe93e88495c371a3f1ba074f5ffd93fe5958247b6feef349f70b`
- **Checkpoint Isolation**: Verified that the production checkpoint remains untouched.

### 2.6 Performance Reconciliation with Canonical JSON Records
Cross-checking reported numbers against `outputs/experiments/phase11_4/cross_fold_comparison.json` and `fold_metrics.json`:

| Metric | Phase 11.4 Baseline JSON | Phase 11.4 Exp B (Selected) JSON |
|---|---|---|
| **Pooled Macro-F1** | 99.09% | **99.81%** |
| **Pooled Accuracy** | 98.91% | **99.77%** |
| **Fold 1 (Unseen 1 HP) F1** | 100.00% | **100.00%** |
| **Fold 2 (Unseen 2 HP) F1** | 100.00% | **99.43%** (16 errors) |
| **Fold 3 (Unseen 3 HP) F1** | 97.26% | **100.00%** (0 errors) |
| **Rec 172 (3 HP IR) Acc** | 67.09% (77 errors) | **100.00% (0 errors)** |
| **Total Trainable Parameters** | 73,156 | **89,604** |
| **Median Inference Latency** | 1.10 ms (model forward) | **1.20 ms** (model) / **2.56 ms** (end-to-end) |

*Audit Correction*: Fold 3 achieved 100% accuracy in Experiment B (completely eliminating errors on 3 HP recording 172), while the 16 remaining errors in Experiment B occurred in Fold 2 (2 HP held out, accuracy 99.32%, Macro-F1 99.43%). Previous prose had transposed Folds 2 and 3.

### 2.7 Automated Test Function Reconciliation
- **Total Test Files**: 27 test modules in `tests/`.
- **Total Test Functions**: Exactly **224 test functions**, all passing.
- **`tests/test_final_model.py` Functions**: Exactly 9 test functions covering artifacts, checksums, isolation, parameters, inference contracts, explainability, and latency.

---

## 3. Audit Sign-off

The repository state has been reconciled and verified against physical disk assets. All subsequent Phase 12 documentation and the public README reflect these canonical, mathematically verified numbers.


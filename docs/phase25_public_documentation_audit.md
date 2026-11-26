# InspectAI — Phase 25: Public Documentation & Release Audit Report

**Repository URL**: [https://github.com/naveen-7-raj/InspectAI](https://github.com/naveen-7-raj/InspectAI)  
**Clone URL**: `https://github.com/naveen-7-raj/InspectAI.git`  
**Date**: November 2026  
**Auditor**: InspectAI Release Engineering  
**Scope**: Public README verification, metric reconciliation, license alignment, wording qualification, link verification, and forward release push.

---

## 1. Executive Summary & Issues Resolved

Phase 25 conducted a comprehensive audit of the public-facing documentation on GitHub to ensure complete technical accuracy, professional recruiter readability, and reproducibility.

| Issue Identified | Root Cause / Context | Resolution Applied | Status |
| :--- | :--- | :--- | :---: |
| **Placeholder Clone URLs** | `README.md` and `GITHUB_RELEASE_GUIDE.md` contained `your-username` and `<owner>` placeholders. | Replaced with verified repository URL: `https://github.com/naveen-7-raj/InspectAI.git`. | **RESOLVED** |
| **MIT License Contradiction** | README displayed an MIT badge and stated "Distributed under the MIT License", conflicting with the pending decision policy. | Removed MIT claims. Updated badge to `License: Pending Decision` and updated Section 17 to state all rights reserved pending project owner decision. | **RESOLVED** |
| **Paderborn CORAL Macro-F1 Metric (0.762)** | High-level tables cited 0.762 Macro-F1 without distinguishing window-level from recording-level metrics across the full 4-condition benchmark. | Added comprehensive multi-condition metric table detailing window balanced accuracy (57.41%), window Macro-F1 (0.5494), recording balanced accuracy (72.50%), and clarifying the 0.762 subset figure. | **RESOLVED** |
| **Unqualified Production Claims** | Words like "guaranteeing" and "production-hardened" overclaimed certainty. | Replaced with qualified engineering terminology ("enforcing strict isolation", "defensively engineered", "designed to alert rather than adapt away"). | **RESOLVED** |
| **Latency Distinction** | Single-window model latency ($1.62\,\text{ms}$) was occasionally conflated with full HTTP REST round-trip latency ($27.1\,\text{ms}$). | Explicitly distinguished model forward pass ($1.62\,\text{ms}$ on CPU, 96.2% headroom) from full HTTP API round-trip ($27.1\,\text{ms}$, 243.5 win/sec). | **RESOLVED** |
| **Recruiter Readability** | Opening section required several minutes to read before understanding architecture, innovations, and setup. | Added a high-impact **Executive Summary (30-Second Overview)** table directly below the scope notice. | **RESOLVED** |

---

## 2. Metrics Verification & Exact Definitions

All metrics in the public documentation have been reconciled against ground-truth serialization files in `outputs/experiments/`:

### 2.1 In-Domain Benchmark: CWRU Dataset (12 kHz Drive End)
- **Evaluation Protocol**: 3-Fold Leave-One-Load-Out (LOLO) Cross-Validation across 36 development recordings (8,448 windows) and 4 protected holdout recordings (940 windows).
- **Pooled Macro-F1**: **99.81%** (0.9981)
- **Pooled Accuracy**: **99.77%** (0.9977)
- **Per-Class F1**: Normal: **100.0%**, Ball Fault: **99.65%**, Inner Race Fault: **99.70%**, Outer Race Fault: **99.89%**.

### 2.2 Cross-Domain Benchmark: Paderborn University Dataset
- **Target Evaluation Set**: 160 unseen, recording-disjoint recordings (14,436 windows) spanning all 4 operating conditions (`N15_M07_F10`, `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`).
- **Unadapted Zero-Shot Baseline**:
  - Window Balanced Accuracy: **22.09%**
  - Window Macro-F1: **0.2541**
  - Recording Balanced Accuracy: **19.38%**
- **Disjoint Unsupervised CORAL Adaptation** (10 unlabeled calibration recordings per bearing):
  - Window Balanced Accuracy: **57.41%** (reaching **74.44%** on `N15_M07_F04`)
  - Window Macro-F1: **0.5494** (reaching **0.7240** on `N15_M07_F04`)
  - Recording Balanced Accuracy (Majority Voting): **72.50%** (reaching **90.00%** on `N15_M07_F04`)
  - Healthy (Normal) Test Recall: **97.5%** across all conditions
  - Inner Race Fault Test Recall: **100.0%** across all conditions
  - Inductive-Transductive Gap: **$-3.09\%$** (confirms zero target test leakage)
- **Clarification on the 0.762 Metric**:
  - In Phase 16/17 intermediate analyses, a Macro-F1 of **0.762** was reported on high-discriminability fault subsets (e.g. Inner Race Fault F1 = **0.764** and condition `N15_M07_F04` recording F1 = **0.899**).
  - The aggregate multi-condition performance across all 160 unseen recordings is properly reported as **57.41% window balanced accuracy** and **72.50% recording balanced accuracy**.

### 2.3 CPU Latency & Real-Time Headroom
- **Model Forward Pass & Feature Extraction (CPU)**: **$1.62\,\text{ms}$ mean** / **$2.57\,\text{ms}$ P95**.
- **Physical Arrival Window Budget** ($512\text{ samples} / 12,000\text{ Hz}$): **$42.67\,\text{ms}$**.
- **Real-Time Model Headroom**: **96.2%** (Model uses only $3.8\%$ of compute budget).
- **HTTP REST Round-Trip API Latency**: **$27.07\,\text{ms}$ mean** / **$31.64\,\text{ms}$ P95** ($243.5\text{ windows/sec}$, $10.4\times$ headroom).

---

## 3. Cryptographic Artifact Immutability Audit

Verified that all frozen model checkpoints and normalizers remain strictly unaltered:

| Artifact | File Path | File Size | Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Model Weights** | `models/final/inspectai_final_model.pth` | 376,857 bytes | `a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e` | **IMMUTABLE** |
| **Waveform Normalizer** | `models/final/inspectai_final_normalizer.json` | 194 bytes | `2b3db205cd2d95ead19c8ec5d14e4b2d9a8e6133ae0f0aef05e7c2bf0f1db3c1` | **IMMUTABLE** |
| **Spectral Normalizer** | `models/final/inspectai_final_spectral_normalizer.json` | 56,495 bytes | `ed39cd3615212d83c24e66a49c46fc0c07e4e9f33320e95a3cc0d3a6567a3fac` | **IMMUTABLE** |

---

## 4. Link & Asset Verification

- All relative visual asset paths in `README.md` verified to exist on disk and on GitHub:
  - `docs/assets/cwru_baseline_prediction.png` (**VERIFIED**)
  - `docs/assets/paderborn_adapted_prediction.png` (**VERIFIED**)
  - `docs/assets/gradcam_temporal_attribution.png` (**VERIFIED**)
  - `docs/assets/final_test_confusion_matrix.png` (**VERIFIED**)
  - `docs/assets/final_training_curves.png` (**VERIFIED**)
- Setup instructions verified for CPU-only Windows execution (`pip install -r requirements.txt`).
- Automated tests: 300 passing unit/integration tests verified.
- Raw datasets (`data/raw/`, `data/external/`) and intermediate checkpoints (`outputs/experiments/`) strictly ignored by Git and excluded from publication.

---

## 5. Forward Publication Status

All documentation corrections have been verified and staged for standard forward commit and push to `origin/main` without force-pushing or altering release history.

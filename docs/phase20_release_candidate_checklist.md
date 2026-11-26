# InspectAI — Phase 20: Release Candidate Checklist

**Project**: InspectAI – Explainable Industrial Machine Health Monitoring System  
**Release Candidate**: Production `SpectralFusionCNN` Pipeline  
**Target Checkpoint**: `models/final/inspectai_final_model.pth`  
**SHA-256**: `a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e`  
**Evaluation Date**: November 2026  
**Status**: APPROVED FOR PORTFOLIO RELEASE  

---

## 1. Cryptographic Artifact Freeze & Model Provenance

| Verification Item | Requirement / Contract | Verified Value / Status | Result |
| :--- | :--- | :--- | :---: |
| **Model Weights Checkpoint** | Checkpoint exists and matches production SHA-256 | `a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e` | **PASS** |
| **Waveform Normalizer** | Frozen JSON file with verified training statistics | Mean: `-3.2797e-06`, Std: `0.3469` (`models/final/inspectai_final_normalizer.json`) | **PASS** |
| **Spectral Normalizer** | Frozen log-spectral FFT normalizer | Min: `-14.6452`, Max: `5.6430` (`models/final/inspectai_final_spectral_normalizer.json`) | **PASS** |
| **Architecture Contract** | Dual-branch 1D CNN + FFT Spectral Fusion ($d=576$) | `SpectralFusionCNN` with 89,604 parameters on CPU | **PASS** |
| **Model Immutability Rule** | No gradient updates or fine-tuning on external data | Weights remain 100% frozen; zero test-time backpropagation | **PASS** |

---

## 2. Scientific Integrity & Data Governance

| Verification Item | Requirement / Contract | Verified Value / Status | Result |
| :--- | :--- | :--- | :---: |
| **Recording-Level Split** | Zero overlap between adaptation and evaluation recordings | 160 adaptation vs 160 evaluation recordings (Overlap $= 0$) | **PASS** |
| **Label Isolation** | Unsupervised adaptation must never access target labels | `DomainAdapter` methods accept only unlabeled feature matrix $X$ | **PASS** |
| **Threshold Provenance** | Thresholds derived from physical / statistical theory | $\tau_{\text{drift}} = 0.40$, Kurtosis $= 4.5$, $P(\text{Fault}) = 0.35$ (zero label leakage) | **PASS** |
| **Historical Honesty** | Zero-shot failure on Paderborn transparently documented | Reported honestly in Phase 14 ($F_1 \approx 0.25$) & Phase 15 root cause | **PASS** |
| **Generalization Claim** | CORAL adaptation validated on unseen recordings | Documented macro $F_1 \approx 0.76$ on recording-disjoint test split | **PASS** |

---

## 3. Application & Deployment Verification

| Verification Item | Requirement / Contract | Verified Value / Status | Result |
| :--- | :--- | :--- | :---: |
| **REST Health Probes** | `/health` and `/api/v1/health` return HTTP 200 | HTTP 200 OK, `model_loaded=True`, `device="cpu"` | **PASS** |
| **Baseline Prediction** | Standard 2,048-sample vibration inference | HTTP 200 OK, outputs predicted class, probabilities, latency | **PASS** |
| **Adaptive Prediction** | Uncalibrated graceful fallback to baseline | Output `is_adapted=False`, `inference_mode="adaptive_fallback"` | **PASS** |
| **Real External Validation** | Successful adaptation on real Paderborn file | Verified on `N09_M07_F10_K001_1.mat`, $\kappa = 35,481.5$, `is_adapted=True` | **PASS** |
| **WebSocket Streaming** | Real-time simulated vibration replay over WS | Active connection, `event="prediction"`, 50x speed supported | **PASS** |
| **Prometheus Exposition** | `/metrics` serves standard Prometheus format | Counters, histograms, condition number, drift gauges verified | **PASS** |
| **Docker Specifications** | Hardened Dockerfile and Compose orchestration | Non-root `appuser`, CPU wheel, `:ro` volumes, healthchecks | **PASS** |

---

## 4. Industrial Safety Guardrails & Drift Monitor

| Verification Item | Requirement / Contract | Verified Value / Status | Result |
| :--- | :--- | :--- | :---: |
| **Fault Masking Prevention** | Never adapt away potential machine fault indications | Recalibrations during fault $= 0$; strictly enforced | **PASS** |
| **Impulsive Kurtosis Gate** | Suppress adaptation when Kurtosis $> 4.5$ | Kurtosis $= 67.4 \rightarrow$ `SUPPRESSED_FAULT_SUSPECT` | **PASS** |
| **Posterior Defect Gate** | Suppress adaptation when $P(\text{Fault}) \ge 0.35$ | Defect posterior $= 0.75 \rightarrow$ `SUPPRESSED_FAULT_SUSPECT` | **PASS** |
| **Conditioning Gate** | Reject ill-conditioned shadow covariance matrices | Candidate rejected if $\kappa(C_T) > 10^7$ or singular | **PASS** |
| **Cooldown Window** | Prevent rapid consecutive fluttering recalibrations | Minimum 30 windows enforced (`COOLDOWN_ACTIVE`) | **PASS** |
| **Instant Rollback** | Restore previous valid adapter upon detected anomaly | `monitor.rollback()` succeeds in $O(1)$, restoring active adapter | **PASS** |

---

## 5. Performance SLAs & Resource Footprints

| Verification Item | Requirement / SLA | Measured Value | Result |
| :--- | :--- | :--- | :---: |
| **Baseline Latency (Mean)** | $\le 50.0\,\text{ms}$ per window | **$28.73\,\text{ms}$** (P95: $38.84\,\text{ms}$) | **PASS** |
| **Adaptive Latency (Mean)** | $\le 50.0\,\text{ms}$ per window | **$27.07\,\text{ms}$** (P95: $31.64\,\text{ms}$) | **PASS** |
| **Adaptation Overhead** | $\le 5.0\,\text{ms}$ overhead | Negligible / negative within jitter ($-1.66\,\text{ms}$) | **PASS** |
| **Streaming Throughput** | $\ge 50$ windows/sec engine capacity | **$243.5$ windows/sec** | **PASS** |
| **Real-Time Headroom** | $\ge 4.0\times$ headroom over $23.4$ win/sec rate | **$10.4\times$ real-time headroom** | **PASS** |
| **Process RSS Memory** | $\le 2,000\,\text{MB}$ total process memory | **$672.3\,\text{MB}$** (well within 4GB container limit) | **PASS** |
| **API Error Rate** | $0.0\%$ unhandled 500 errors | $0.0\%$ (200/200 requests returned 200 OK) | **PASS** |

---

## 6. Documented Operational Context & Known Limitations

1. **Simulated Streaming Replay**: The WebSocket streaming interface replays authentic laboratory-recorded vibration files from the CWRU and Paderborn testbeds. It faithfully simulates edge sensor arrival intervals ($42.6\,\text{ms}$ at $1\times$ speed). Connection to physical sensor DAQ hardware (e.g. NI-DAQmx or industrial Modbus/OPC-UA) requires an edge driver adapter.
2. **Initial Calibration Assumption**: Unsupervised CORAL adaptation requires 25–50 windows of running machine data collected under normal operating baseline. If a machine is commissioned already severely defective during initial calibration, baseline mode should be used.
3. **Host Docker Engine**: On development workstations where the Docker Desktop daemon is inactive, container configurations have been verified via automated AST inspection, Compose YAML validation, and in-process test harnesses.

---

## 7. Final Release Sign-Off

- **Cryptographic Model Hash**: `a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e` (VERIFIED)
- **Automated Tests**: 100% Passing (289/289 Unit & Integration Tests)
- **Deployment Status**: Production Hardened, Fully Observable, Guardrailed, and Portfolio Ready.

**Verdict**: **RECOMMENDED FOR PRODUCTION CANDIDATE & PORTFOLIO RELEASE**

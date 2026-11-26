# InspectAI Phase 14: Real-Data Acquisition Report

## 1. Executive Summary

Phase 14 executes the first empirical acquisition and validation of external bearing vibration data for InspectAI using a 4-bearing real-data pilot from the **Paderborn University (PU) Bearing Dataset**.

All four pilot RAR archives were acquired directly from the official Zenodo archival repository (DOI: `10.5281/zenodo.15845309`), cryptographically audited, and fully extracted into individual experimental directories under `data/external/extracted/`.

```
data/external/
├── raw/
│   ├── K001.rar  (173,881,721 bytes) [Normal / Healthy Reference]
│   ├── KA04.rar  (180,920,703 bytes) [Outer Race Fault - Real Fatigue]
│   ├── KI04.rar  (172,644,279 bytes) [Inner Race Fault - Real Fatigue]
│   └── KB23.rar  (170,933,916 bytes) [Ball Fault - Combined Real Damage]
└── extracted/
    ├── K001/  (80 .mat files)
    ├── KA04/  (80 .mat files)
    ├── KI04/  (80 .mat files)
    └── KB23/  (80 .mat files)
```

Total raw archive payload: **698,380,619 bytes** (~666.04 MB), containing **320 individual measurement files** across 4 operating conditions.

---

## 2. Dataset Provenance & Attribution

- **Dataset Title**: KAT-DataCenter Bearing Data Set
- **Institution**: Chair of Design and Drive Technology (KAt), Paderborn University, Germany
- **Authors**: Christian Lessmeier, James Kurths, Detmar Zimmer et al.
- **Official DOI**: `10.5281/zenodo.15845309`
- **Citation**: C. Lessmeier, J. K. Kimotho, D. Zimmer, "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signature of Electric Motors: A Benchmark Data Set", *PHM Society Conference*, 2016.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 3. Rectification of Phase 13 Documentation Errors

Before executing the acquisition pipeline, a comprehensive audit of Phase 13 documentation was performed, identifying and resolving two critical discrepancies:

### 3.1 Taxonomy Discrepancy & `KA04` Duplicate Classification
- **Phase 13 Error**: In `docs/phase13_external_dataset_research.md`, bearing `KA04` was erroneously listed in two contradictory roles: both as an Inner Race candidate and an Outer Race candidate.
- **Authoritative Resolution**: In Paderborn nomenclature:
  - `K0` = Healthy reference bearing (`K001` run-in > 50h)
  - `KA` = **Außenring** (Outer ring damage; `KA04` = accelerated lifetime fatigue spalling)
  - `KI` = **Innenring** (Inner ring damage; `KI04` = accelerated lifetime fatigue spalling)
  - `KB` = **Kombiniert / Wälzkörper** (Combined damage / rolling element flaking; `KB23` = fatigue damage on balls)
  The duplicate classification has been completely eliminated across all manifests and documentation.

### 3.2 Archive Download Size Discrepancy
- **Phase 13 Error**: Reported download size was estimated at ~75 MB for the pilot subset.
- **Authoritative Measurement**: Each Paderborn archive is compressed as a RAR archive containing 80 complete high-resolution 64 kHz multi-channel MATLAB v5/v7 files. The actual archive sizes range between 163.0 MB and 172.5 MB each. Total pilot acquisition is **666.04 MB**, not 75 MB.

---

## 4. Cryptographic Acquisition Manifest

| Bearing ID | Target Class | Damage Mechanism | Archive File | File Size (Bytes) | SHA-256 Checksum | MAT Files |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K001** | Normal | Healthy Reference (>50h run-in) | `K001.rar` | 173,881,721 | `0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5` | 80 |
| **KA04** | Outer Race | Accelerated Lifetime Fatigue (Spall) | `KA04.rar` | 180,920,703 | `6fc317157e2cd4cb55bc54d6fac1b1753f8ab31abcccd1ac68cbfb78e5e8299d` | 80 |
| **KI04** | Inner Race | Accelerated Lifetime Fatigue (Spall) | `KI04.rar` | 172,644,279 | `8fde861a9ceac4b41fe8a030b4c5bbb4b9b9af0590d9338a8de0150b4661be77` | 80 |
| **KB23** | Ball Fault | Combined Fatigue Flaking on Balls | `KB23.rar` | 170,933,916 | `a07e976b285438c33492650c4f5b28250a49e7855990191b30b6bcb541831c6c` | 80 |

The structured machine-readable manifest is stored at `outputs/experiments/phase14/acquisition_manifest.json`.

---

## 5. Security, Isolation & Git Exclusions

1. **Zero Data Leakage / Contamination**: Raw and extracted Paderborn data are isolated under `data/external/`. No external signals have been ingested into CWRU training folds, validation sets, or hyperparameter selection pipelines.
2. **Git Exclusion Guarantee**: `.gitignore` explicitly rules out:
   - `data/external/raw/`
   - `data/external/extracted/`
   - `*.mat` and `*.rar` files
   Ensuring zero repository bloat or accidental license distribution.
3. **ID Collision Immunity**: All external bearings use alphanumeric IDs (`K001`, `KA04`, `KI04`, `KB23`) prefixed with `EXT_PU_` during loader ingestion, preventing any collision with CWRU numeric IDs (`105`, `118`, etc.) or historical quarantined benchmarks (`97`, `169`, `185`, `197`).

# InspectAI Phase 14: Real MATLAB File Inspection Report

## 1. Executive Summary

To eliminate guesswork and avoid reliance on synthetic test fixtures, direct binary inspection was executed across the downloaded Paderborn University `.mat` files using `scipy.io.loadmat`.

Representative recordings across all four pilot bearings were parsed and analyzed:
- `K001`: Healthy Baseline Reference
- `KA04`: Accelerated Lifetime Fatigue Outer Race Fault
- `KI04`: Accelerated Lifetime Fatigue Inner Race Fault
- `KB23`: Combined Lifetime Fatigue Ball Fault

---

## 2. File Organization & Operating Condition Encoding

Each extracted directory contains 80 MATLAB files conforming to a standard naming syntax:

$$\text{N\langle Speed\rangle\_M\langle Torque\rangle\_F\langle Force\rangle\_\langle BearingID\rangle\_\langle MeasurementIndex\rangle.mat}$$

For example: `N15_M07_F10_K001_1.mat` encodes:
- **Speed (N)**: `N15` $\to$ 1,500 RPM nominal rotational frequency ($f_r = 25.0\text{ Hz}$).
- **Torque (M)**: `M07` $\to$ 0.7 Nm mechanical load torque (`M01` = 0.1 Nm).
- **Radial Force (F)**: `F10` $\to$ 1,000 N radial load (`F04` = 400 N).
- **Bearing ID**: `K001`.
- **Measurement Index**: 1 to 20 repeated trials per operating condition.

Every physical bearing has 4 operating condition regimes $\times$ 20 measurements = **80 files per bearing**.

---

## 3. MATLAB Internal Structure

All inspected files are formatted as **MATLAB Level 5.0 (v7)** structures. No HDF5 / MATLAB v7.3 parsing library (`h5py` / `mat73`) was required.

The internal structure consists of a top-level struct matching the recording filename:
```
Root Struct: N15_M07_F10_K001_1
├── Description: Character metadata string
├── Info: Measurement run settings and date
├── X: Timebase struct (Raster = 1.5625e-05 seconds -> 64,000 Hz)
└── Y: Struct array containing 7 synchronous sensor channels
```

### Channel Mapping Inside `Y`

| Channel Index | Channel Name | Signal Type | Sample Count | Sampling Rate (Hz) | Physical Unit |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **0** | `force` | Radial test load | 16,001 | 4,000 | N |
| **1** | `phase_current_1` | Motor stator current Phase 1 | 256,001 | 64,000 | A |
| **2** | `phase_current_2` | Motor stator current Phase 2 | 256,001 | 64,000 | A |
| **3** | `speed` | Shaft rotational tachometer | 16,001 | 4,000 | RPM |
| **4** | `temp_2_bearing_module`| Thermocouple temperature | 5 | 1 | °C |
| **5** | `torque` | Torsional reaction load | 16,001 | 4,000 | Nm |
| **6** | `vibration_1` | **Piezoelectric Accelerometer** | **256,001** | **64,000** | $\text{m/s}^2$ |

**Critical Finding**: The primary vibration signal is strictly situated at **Channel Index 6** (`vibration_1`), sampled continuously at **64,000 Hz** for exactly **4.000015 seconds** (256,001 samples).

---

## 4. Empirical Signal Statistics (Channel 6: `vibration_1`)

Representative signal metrics measured at standard operating condition (`N15_M07_F10`):

| Bearing ID | Fault Class | Min ($\text{m/s}^2$) | Max ($\text{m/s}^2$) | Mean ($\text{m/s}^2$) | Std ($\text{m/s}^2$) | RMS ($\text{m/s}^2$) | Peak ($\text{m/s}^2$) | Crest Factor | Kurtosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K001** | Normal | -4.93 | +4.39 | -0.0188 | 0.4062 | 0.4066 | 4.93 | 12.12 | 13.10 |
| **KA04** | Outer Race | -6.21 | +5.72 | -0.0191 | 0.7725 | 0.7727 | 6.21 | 8.04 | 7.91 |
| **KI04** | Inner Race | -4.54 | +4.61 | -0.0185 | 0.5481 | 0.5484 | 4.61 | 8.41 | 8.08 |
| **KB23** | Ball Fault | -10.37 | +10.15 | -0.0195 | 1.0851 | 1.0853 | 10.37 | 9.55 | 9.77 |

### Engineering Observations:
1. **Physical Acceleration Units**: Signal values range between $-10$ and $+10\text{ m/s}^2$. Dividing by standard gravitational acceleration ($g = 9.80665\text{ m/s}^2$) yields acceleration values in $g$ units:
   - `K001`: $\text{RMS} \approx 0.0415\text{ g}$
   - `KA04`: $\text{RMS} \approx 0.0788\text{ g}$
   - `KI04`: $\text{RMS} \approx 0.0559\text{ g}$
   - `KB23`: $\text{RMS} \approx 0.1107\text{ g}$
2. **DC Sensor Offset**: All recordings exhibit a consistent negative DC sensor bias of approximately $-0.019\text{ m/s}^2$ due to accelerometer conditioning electronics. This confirms the necessity of zero-mean subtraction during preprocessing.
3. **Kurtosis Behavior**: All recordings exhibit high kurtosis (> 7.9), characteristic of mechanical impact dynamics and non-Gaussian vibration bursts.

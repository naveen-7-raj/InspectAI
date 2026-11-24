# Data Dictionary & Fault Physics Specification

## 1. Primary Benchmark: Case Western Reserve University (CWRU) Bearing Dataset

The primary benchmark dataset utilized for training and validating InspectAI is the **CWRU Bearing Data Center** benchmark.

### 1.1 Sensor Channels
| Channel Code | Sensor Placement | Description | Units |
|---|---|---|---|
| `DE_time` | Drive-End Accelerometer | Measures radial acceleration at the drive end bearing housing | $g$ ($m/s^2$) |
| `FE_time` | Fan-End Accelerometer | Measures radial acceleration at the fan end bearing housing | $g$ ($m/s^2$) |
| `BA_time` | Base Accelerometer | Supporting accelerometer mounted on the motor base plate | $g$ ($m/s^2$) |
| `RPM` | Tachometer | Motor operational rotational speed | Revolutions per minute |

### 1.2 Fault Classes & Severities
| Class ID | Target Label | Defect Location | Defect Diameter |
|---|---|---|---|
| `0` | `Normal` | None (Healthy baseline) | 0.000 inches |
| `1` | `Ball_Fault` | Rolling Element (Ball) | 0.007" / 0.014" / 0.021" / 0.028" |
| `2` | `Inner_Race_Fault` | Inner Raceway (BPFI) | 0.007" / 0.014" / 0.021" / 0.028" |
| `3` | `Outer_Race_Fault` | Outer Raceway (BPFO) | 0.007" / 0.014" / 0.021" / 0.028" |

---

## 2. Theoretical Characteristic Fault Frequencies

When a rolling element strikes a localized defect, repetitive transient shock pulses are produced at fundamental characteristic frequencies governed by bearing geometry and shaft rotational speed ($f_r$):

1. **Ball Pass Frequency Outer Race (BPFO)**:
   $$BPFO = \frac{N_b}{2} f_r \left(1 - \frac{d}{D} \cos \alpha \right)$$

2. **Ball Pass Frequency Inner Race (BPFI)**:
   $$BPFI = \frac{N_b}{2} f_r \left(1 + \frac{d}{D} \cos \alpha \right)$$

3. **Ball Spin Frequency (BSF)**:
   $$BSF = \frac{D}{2d} f_r \left(1 - \left(\frac{d}{D} \cos \alpha\right)^2 \right)$$

4. **Fundamental Train Frequency / Cage Frequency (FTF)**:
   $$FTF = \frac{1}{2} f_r \left(1 - \frac{d}{D} \cos \alpha \right)$$

*Where:*
- $N_b$: Number of rolling elements
- $d$: Ball diameter
- $D$: Pitch diameter
- $\alpha$: Contact angle
- $f_r$: Shaft speed in Hz ($\text{RPM} / 60$)


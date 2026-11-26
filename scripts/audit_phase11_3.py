import json
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path

print("=== 1. SIGNAL STATISTICS AUDIT ===")
df = pd.read_csv("outputs/experiments/phase11_3/recording_signal_statistics.csv")
subset = df[df["recording_id"].isin([170, 171, 172, 108, 212, 188])][
    ["recording_id", "fault_class", "load_hp", "rms_amplitude", "peak_amplitude", "kurtosis", "crest_factor", "high_band_ratio"]
]
print(subset.to_string(index=False))

print("\n=== 2. KINEMATIC BEARING FREQUENCIES AUDIT ===")
# CWRU Drive End: SKF 6205-2RS JEM
# d = 0.3126 in (7.94 mm), D = 1.537 in (39.04 mm), Nb = 9, contact angle = 0 deg
d = 7.94
D = 39.04
Nb = 9
gamma = d / D  # 0.203381
print(f"Bearing parameters: d={d}mm, D={D}mm, Nb={Nb}, gamma={gamma:.4f}")

# CWRU Drive end standard kinematic multipliers:
# BPFI multiplier = (Nb/2)*(1 + gamma) = 4.5 * 1.20338 = 5.4152
# BPFO multiplier = (Nb/2)*(1 - gamma) = 4.5 * 0.79662 = 3.5848
# BSF multiplier = (D/(2*d))*(1 - gamma^2) = (39.04/15.88)*(1 - 0.04136) = 2.4584 * 0.95864 = 2.3567
# FTF multiplier = 0.5 * (1 - gamma) = 0.3983
print(f"Multipliers: BPFI/fr={Nb/2*(1+gamma):.4f}, BPFO/fr={Nb/2*(1-gamma):.4f}, BSF/fr={(D/(2*d))*(1-gamma**2):.4f}, FTF/fr={0.5*(1-gamma):.4f}")

rpms = {"0 HP": 1797, "1 HP": 1772, "2 HP": 1750, "3 HP (rec 172)": 1730, "3 HP (nom)": 1725}
for name, rpm in rpms.items():
    fr = rpm / 60.0
    bpfi = (Nb / 2.0) * fr * (1 + gamma)
    bpfo = (Nb / 2.0) * fr * (1 - gamma)
    bsf = (D / (2.0 * d)) * fr * (1 - gamma**2)
    ftf = (1.0 / 2.0) * fr * (1 - gamma)
    print(f"{name} ({rpm} RPM, fr={fr:.2f} Hz): BPFI={bpfi:.2f} Hz, BPFO={bpfo:.2f} Hz, BSF={bsf:.2f} Hz, FTF={ftf:.2f} Hz")

print("\n=== 3. SOFTMAX MARGIN AUDIT ===")
df_misc = pd.read_csv("outputs/experiments/phase11_3/misclassification_analysis.csv")
print("Total windows in 172:", len(df_misc))
misc_wins = df_misc[~df_misc["is_correct"]]
corr_wins = df_misc[df_misc["is_correct"]]
print("Misclassified windows count:", len(misc_wins))
print("Misclassified mean prob_ball:", misc_wins["prob_ball"].mean())
print("Misclassified mean prob_inner:", misc_wins["prob_inner_race"].mean())
print("Misclassified mean winning margin:", misc_wins["winning_margin"].mean())
print("Correct mean winning margin:", corr_wins["winning_margin"].mean())

print("\n=== 4. STATISTICAL SIGNIFICANCE (p-value) AUDIT ===")
# In Phase 11.2, evaluation errors:
# Fold 1 (1 HP): 0 / 2348
# Fold 2 (2 HP): 0 / 2342
# Fold 3 (3 HP): 62 / 2348
# Window-level contingency table:
_, p_fisher_windows = scipy.stats.fisher_exact([[0, 4690], [62, 2286]])
print(f"Window-level Fisher exact test p-value: {p_fisher_windows:.4e}")
# Recording-level contingency table:
# 1 HP: 0 / 10 recordings failed
# 2 HP: 0 / 10 recordings failed
# 3 HP: 1 / 10 recordings failed
_, p_fisher_recs = scipy.stats.fisher_exact([[0, 20], [1, 9]])
print(f"Recording-level Fisher exact test p-value: {p_fisher_recs:.4f}")

print("\n=== 5. EARLY STOPPING EPOCH FOR FOLD 3 AUDIT ===")
with open("outputs/experiments/phase11_2/training_history.json") as f:
    h = json.load(f)

for fold_k in h:
    fh = h[fold_k]
    hist = fh["history"]
    best_ep = fh["selected_best_epoch"]
    print(f"--- {fold_k} (selected best epoch: {best_ep}, total epochs trained: {len(hist['train_loss'])}) ---")
    for ep in range(len(hist["train_loss"])):
        print(f"  Epoch {ep+1:2d}: train_loss={hist['train_loss'][ep]:.4f}, val_loss={hist['val_loss'][ep]:.4f}, val_macro_f1={hist['val_macro_f1'][ep]:.4f}")


with open("outputs/experiments/phase11_2/fold_metrics.json") as f:
    fm = json.load(f)
for k in fm:
    print(f"{k}: best_epoch={fm[k].get('best_epoch')}, final_epoch={fm[k].get('final_epoch')}")


print("\n=== 6. GRAD-CAM ATTRIBUTION AUDIT ===")
with open("outputs/experiments/phase11_3/analysis_summary.json") as f:
    summary = json.load(f)
gradcam_cases = summary["gradcam"]["cases"]
for c in gradcam_cases:
    print(f"Case: {c['case_name']} | Pred: {c['predicted_class']} | Concentration: {c['attribution_concentration_ratio']}")

print("\n=== 7. SPECTRAL ENERGY RATIO DEFINITIONS ===")
from src.features.spectral_features import compute_spectral_band_energies
# Let's inspect how compute_spectral_band_energies defines bands
import inspect
print(inspect.getsource(compute_spectral_band_energies))

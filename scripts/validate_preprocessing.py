"""Script to validate external signal preprocessing on real Paderborn measurements.

Verifies:
1. Physical unit conversion (m/s^2 -> g).
2. Polyphase rational resampling (64 kHz -> 12 kHz, up=3, down=16).
3. DC offset removal and Butterworth bandpass filtering (20 - 5000 Hz).
4. Window segmentation (2,048 samples, 512 step, 75% overlap).
5. Dynamic Hilbert envelope spectrum computation (1025 frequency bins).
6. Application of frozen CWRU normalizers.
7. Verification that no NaN/Inf values exist and dimensions match SpectralFusionCNN requirements.
"""

import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.frequency_features import (
    SpectralNormalizer,
    compute_hilbert_envelope_spectrum,
)
from src.ingestion.external_loader import ExternalDataLoader
from src.preprocessing.external_preprocessor import ExternalPreprocessor
from src.training.normalizer import VibrationNormalizer

EXTRACTED_DIR = REPO_ROOT / "data" / "external" / "extracted"
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments" / "phase14"
MODELS_DIR = REPO_ROOT / "models" / "final"

BEARING_CONFIGS = {
    "K001": {"class": "Normal", "pattern": "N15_M07_F10_K001_1.mat"},
    "KA04": {"class": "Outer Race Fault", "pattern": "N15_M07_F10_KA04_1.mat"},
    "KI04": {"class": "Inner Race Fault", "pattern": "N15_M07_F10_KI04_1.mat"},
    "KB23": {"class": "Ball Fault", "pattern": "N15_M07_F10_KB23_1.mat"},
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = ExternalDataLoader(external_dir=REPO_ROOT / "data" / "external")
    preprocessor = ExternalPreprocessor()

    # Load frozen CWRU normalizers
    waveform_norm_path = MODELS_DIR / "inspectai_final_normalizer.json"
    spectral_norm_path = MODELS_DIR / "inspectai_final_spectral_normalizer.json"

    waveform_norm = VibrationNormalizer.load(waveform_norm_path)
    spectral_norm = SpectralNormalizer.load(spectral_norm_path)
    print(f"Loaded frozen waveform normalizer: mean={waveform_norm.mean:.4f}, std={waveform_norm.std:.4f}")
    print(f"Loaded frozen spectral normalizer: mean range [{np.min(spectral_norm.mean):.4f}, {np.max(spectral_norm.mean):.4f}], n_features={spectral_norm.n_features}")

    results = {}

    for b_id, cfg in BEARING_CONFIGS.items():
        b_dir = EXTRACTED_DIR / b_id
        target_files = list(b_dir.rglob(cfg["pattern"]))
        if not target_files:
            # Fallback to any .mat in the directory
            target_files = list(b_dir.rglob("*.mat"))
        
        if not target_files:
            print(f"No files found for {b_id} in {b_dir}")
            continue

        mat_file = target_files[0]
        print(f"\nProcessing representative file for {b_id} ({cfg['class']}): {mat_file.name}")

        # 1. Load raw recording
        rec = loader.load_paderborn_mat(
            mat_path=mat_file,
            recording_id=mat_file.stem,
            fault_class=cfg["class"],
        )
        raw_signal = rec.signal
        orig_fs = rec.original_sampling_rate
        raw_len = len(raw_signal)

        # 2. Raw signal stats (m/s^2)
        raw_rms = float(np.sqrt(np.mean(raw_signal ** 2)))
        raw_peak = float(np.max(np.abs(raw_signal)))

        # 3. Conditioning: unit conversion (m/s^2 -> g), resampling (64k -> 12k), DC removal, bandpass
        conditioned = preprocessor.condition_continuous_signal(
            raw_signal, orig_fs=orig_fs, units=rec.units
        )
        cond_len = len(conditioned)
        cond_fs = preprocessor.target_fs
        cond_mean = float(np.mean(conditioned))
        cond_rms = float(np.sqrt(np.mean(conditioned ** 2)))
        cond_peak = float(np.max(np.abs(conditioned)))

        # 4. Window segmentation (2048 samples, 512 step)
        windows, win_meta = preprocessor.segment_windows(conditioned)
        num_windows, win_len = windows.shape

        # 5. Envelope spectra (1025 bins)
        _, spectra, _ = compute_hilbert_envelope_spectrum(windows, sampling_rate_hz=cond_fs)
        spec_num, spec_bins = spectra.shape

        # 6. Apply frozen CWRU normalizers
        norm_windows = waveform_norm.transform(windows)
        norm_spectra = spectral_norm.transform(spectra)

        # 7. Check contracts
        has_nan = bool(np.isnan(norm_windows).any() or np.isnan(norm_spectra).any())
        has_inf = bool(np.isinf(norm_windows).any() or np.isinf(norm_spectra).any())

        res_record = {
            "bearing_id": b_id,
            "fault_class": cfg["class"],
            "file_name": mat_file.name,
            "original_sampling_rate_hz": orig_fs,
            "original_samples": raw_len,
            "original_duration_seconds": float(raw_len / orig_fs),
            "original_rms_ms2": raw_rms,
            "original_peak_ms2": raw_peak,
            "target_sampling_rate_hz": cond_fs,
            "conditioned_samples": cond_len,
            "conditioned_duration_seconds": float(cond_len / cond_fs),
            "conditioned_mean_g": cond_mean,
            "conditioned_rms_g": cond_rms,
            "conditioned_peak_g": cond_peak,
            "window_count": num_windows,
            "window_length": win_len,
            "spectral_bins": spec_bins,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "norm_waveform_min": float(np.min(norm_windows)),
            "norm_waveform_max": float(np.max(norm_windows)),
            "norm_waveform_mean": float(np.mean(norm_windows)),
            "norm_waveform_std": float(np.std(norm_windows)),
            "norm_spectral_min": float(np.min(norm_spectra)),
            "norm_spectral_max": float(np.max(norm_spectra)),
            "norm_spectral_mean": float(np.mean(norm_spectra)),
            "contract_valid": bool(
                win_len == 2048
                and spec_bins == 1025
                and not has_nan
                and not has_inf
                and num_windows > 0
            ),
        }
        results[b_id] = res_record

        print(f"  Conditioned signal: {cond_len} samples ({cond_len/cond_fs:.2f} s) at {cond_fs} Hz")
        print(f"  Windows: {num_windows} windows of shape ({win_len},), spectra of shape ({spec_bins},)")
        print(f"  Normalized waveform: range [{res_record['norm_waveform_min']:.2f}, {res_record['norm_waveform_max']:.2f}], mean={res_record['norm_waveform_mean']:.3f}, std={res_record['norm_waveform_std']:.3f}")
        print(f"  Normalized spectra: range [{res_record['norm_spectral_min']:.2f}, {res_record['norm_spectral_max']:.2f}]")
        print(f"  Contract valid: {res_record['contract_valid']}")

    out_file = OUTPUT_DIR / "preprocessing_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved preprocessing validation results to {out_file}")


if __name__ == "__main__":
    main()

"""Unit tests verifying Phase 10.5 Dataset Generalization and Failure Analysis deliverables.

Tests verify:
1. Presence and non-emptiness of all Phase 10.5 experiment artifact files.
2. Cross-experiment metric reconciliation integrity.
3. Recording-level metadata completeness for all 13 recordings.
4. Cross-validation partition integrity, disjointness, and zero test leakage.
5. Presence and validity of all generated diagnostic figures.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "outputs" / "experiments" / "phase10_5"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase10_5"


class TestPhase105GeneralizationAudit:
    """Test suite for Phase 10.5 deliverables and audit integrity."""

    def test_experiment_artifacts_exist(self):
        """Verify all JSON and CSV audit artifacts are present and non-empty."""
        expected_files = [
            "cross_experiment_reconciliation.json",
            "cv_integrity_report.json",
            "generalization_audit_master_summary.json",
            "recording_metadata_table.csv",
            "recording_metadata_table.json",
            "signal_failure_analysis.json",
        ]
        for fname in expected_files:
            fpath = EXP_DIR / fname
            assert fpath.is_file(), f"Missing audit file: {fname}"
            assert fpath.stat().st_size > 0, f"Empty audit file: {fname}"

    def test_figure_artifacts_exist(self):
        """Verify all 6 high-resolution diagnostic figures were generated."""
        expected_figures = [
            "amplitude_distributions.png",
            "envelope_spectra_comparison.png",
            "frequency_band_energy_barchart.png",
            "power_spectral_density.png",
            "severity_confounding_grid.png",
            "time_domain_waveforms.png",
        ]
        for fig_name in expected_figures:
            fig_path = FIG_DIR / fig_name
            assert fig_path.is_file(), f"Missing figure: {fig_name}"
            assert fig_path.stat().st_size > 5000, f"Figure file suspiciously small: {fig_name}"

    def test_recording_metadata_table_integrity(self):
        """Verify recording metadata covers exactly 13 recordings with verified statistics."""
        csv_path = EXP_DIR / "recording_metadata_table.csv"
        df = pd.read_csv(csv_path)

        assert len(df) == 13
        assert set(df["recording_id"].astype(str)) == {
            "97", "98", "99", "100", "105", "118", "130", "169", "185", "197", "209", "222", "234"
        }

        # Check statistical metrics are non-zero and finite
        assert (df["rms_g"] > 0).all()
        assert (df["peak_g"] > 0).all()
        assert (df["crest_factor"] > 0).all()
        assert (df["kurtosis"] > 0).all()

        # Check dev and test partition counts
        assert len(df[df["split"] == "dev"]) == 9
        assert len(df[df["split"] == "test"]) == 4

    def test_cross_validation_integrity_report(self):
        """Verify cross-validation group isolation and zero test leakage."""
        with open(EXP_DIR / "cv_integrity_report.json", "r", encoding="utf-8") as f:
            cv_report = json.load(f)

        assert cv_report["total_windows"] == 2928
        assert cv_report["development_windows"] == 2110
        assert cv_report["test_windows"] == 818
        assert cv_report["test_leakage_detected"] is False

        # Group isolation
        assert len(cv_report["fold_0"]["recording_overlap"]) == 0
        assert len(cv_report["fold_1"]["recording_overlap"]) == 0

        # Exact single evaluation
        assert cv_report["oof_coverage"]["exact_single_evaluation"] is True
        assert cv_report["oof_coverage"]["val_windows_overlap"] == 0
        assert cv_report["oof_coverage"]["uncovered_dev_windows"] == 0

    def test_cross_experiment_reconciliation(self):
        """Verify metric reconciliation numbers across Phase 10.2, 10.3, and 10.4."""
        with open(EXP_DIR / "cross_experiment_reconciliation.json", "r", encoding="utf-8") as f:
            recon = json.load(f)

        assert recon["Phase_10_2"]["trainable_parameters"] == 106244
        assert recon["Phase_10_2"]["oof_macro_f1"] == 0.8378
        assert recon["Phase_10_2"]["oof_accuracy"] == 0.8626

        assert recon["Phase_10_3"]["trainable_parameters"] == 106244
        assert recon["Phase_10_3"]["oof_macro_f1"] == 0.8378
        assert recon["Phase_10_3"]["oof_accuracy"] == 0.8626

        assert recon["Phase_10_4"]["trainable_parameters"] == 73156
        assert recon["Phase_10_4"]["oof_macro_f1"] == 0.8109
        assert recon["Phase_10_4"]["oof_accuracy"] == 0.8469


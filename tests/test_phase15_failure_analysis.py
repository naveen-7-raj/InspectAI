"""Phase 15 Failure Analysis Integration and Verification Tests.

Validates:
1. Correctness of Phase 14 audit calculations and confusion matrix math.
2. Accurate window count reconciliation based on physical sample counts.
3. Pipeline implementation parity (zero implementation bugs).
4. Kinematic defect frequency calculations for 6205 vs 6203 bearings.
5. Integrity and schema of Phase 15 JSON output deliverables.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.run_phase15_failure_analysis import (
    audit_phase14_results,
    compare_pipeline_implementations,
    analyze_spectral_frequency_mismatch,
    KINEMATICS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiments" / "phase15"


class TestPhase15FailureAnalysis:
    """Tests validating the failure analysis investigation."""

    def test_phase14_metrics_audit_consistency(self):
        """Verify that Phase 14 primary accuracy matches exact confusion matrix recalculation."""
        audit = audit_phase14_results()
        rep_acc = audit["primary_reported_accuracy"]
        recalc_acc = audit["primary_recalculated_accuracy"]
        assert abs(rep_acc - recalc_acc) < 1e-6
        assert audit["primary_reported_windows"] == 7211

    def test_window_count_reconciliation_math(self):
        """Verify that window count per file follows floor((N - 2048) / 512) + 1."""
        audit = audit_phase14_results()
        recon = audit["window_count_reconciliation"]
        assert recon["actual_windows_extracted"] == 7211
        assert recon["expected_nominal_at_90_per_file"] == 7200
        assert recon["discrepancy"] == 11

        # Check breakdown
        for b_id, b_info in recon["file_window_breakdown"].items():
            assert b_info["num_files"] == 20
            assert b_info["total_windows"] in (1801, 1803, 1804)

    def test_pipeline_comparison_zero_bugs(self):
        """Verify that side-by-side pipeline audit confirms zero implementation bugs."""
        pipeline_audit = compare_pipeline_implementations()
        assert pipeline_audit["implementation_bug_detected"] is False
        for step, check in pipeline_audit["pipeline_comparison_matrix"].items():
            assert check["mismatch"] is False, f"Unexpected mismatch in {step}"

    def test_kinematic_frequency_calculations(self):
        """Verify kinematic defect frequencies for CWRU (6205) and Paderborn (6203)."""
        cwru = KINEMATICS["CWRU_6205_1797RPM"]
        pu = KINEMATICS["PADERBORN_6203_1500RPM"]

        # BPFO = (n/2) * fr * (1 - d/Dp)
        expected_cwru_bpfo = (9 / 2) * 29.95 * (1 - 7.94 / 39.04)
        assert abs(cwru["bpfo_hz"] - expected_cwru_bpfo) < 0.2

        expected_pu_bpfo = (8 / 2) * 25.0 * (1 - 6.75 / 28.5)
        assert abs(pu["bpfo_hz"] - expected_pu_bpfo) < 0.2

        # Frequency gap
        spectral_audit = analyze_spectral_frequency_mismatch()
        diffs = spectral_audit["theoretical_discrepancies"]
        assert diffs["outer_race_bpfo_shift_hz"] < -30.0  # ~ -31.04 Hz
        assert diffs["inner_race_bpfi_shift_hz"] < -35.0  # ~ -38.51 Hz

    def test_phase15_deliverables_exist_and_valid(self):
        """Verify that phase15 JSON deliverables are populated and well-formed."""
        fa_path = OUTPUT_DIR / "failure_analysis.json"
        ds_path = OUTPUT_DIR / "domain_shift_comparison.json"

        assert fa_path.exists(), f"Missing deliverable: {fa_path}"
        assert ds_path.exists(), f"Missing deliverable: {ds_path}"

        with open(fa_path, "r", encoding="utf-8") as f:
            fa_data = json.load(f)
        assert "section_1_results_audit" in fa_data
        assert "section_2_pipeline_audit" in fa_data
        assert "section_3_model_activation_audit" in fa_data
        assert "categorized_findings" in fa_data

        with open(ds_path, "r", encoding="utf-8") as f:
            ds_data = json.load(f)
        assert "kinematics" in ds_data
        assert "frequency_discrepancies" in ds_data

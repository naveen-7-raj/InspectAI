"""Tests for Phase 17: CORAL Validation & Generalization Audit.

Verifies:
1. Recording-disjoint split manifest structure, counts, and zero-overlap invariants.
2. Unsupervised CORAL mathematical properties (covariance matching, mean alignment, label isolation).
3. Phase 16 baseline and CORAL reproducibility checks.
4. Window-to-recording aggregation logic (majority voting, probability averaging, agreement metrics).
5. Output JSON artifact schema and content validity across all four operating conditions.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.run_phase17_experiments import (
    apply_coral_projection,
    fit_coral_operator,
    aggregate_recording_predictions,
    parse_recording_number,
    REPO_ROOT,
    OUTPUT_DIR,
)


class TestPhase17RecordingSplitManifest:
    """Verify recording-disjoint split manifest properties and invariants."""

    @pytest.fixture
    def manifest(self):
        manifest_path = OUTPUT_DIR / "recording_split_manifest.json"
        assert manifest_path.exists(), f"Missing split manifest: {manifest_path}"
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_all_four_conditions_present(self, manifest):
        expected_conds = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
        assert "conditions" in manifest
        for cond in expected_conds:
            assert cond in manifest["conditions"], f"Condition {cond} missing from manifest"

    def test_recording_counts_per_condition(self, manifest):
        for cond, cond_data in manifest["conditions"].items():
            assert len(cond_data["adaptation_recordings"]) == 40
            assert len(cond_data["evaluation_recordings"]) == 40
            for b_id, b_counts in cond_data["recordings_per_bearing"].items():
                assert b_counts["adaptation_count"] == 10
                assert b_counts["evaluation_count"] == 10

    def test_strict_recording_disjointness_no_overlap(self, manifest):
        for cond, cond_data in manifest["conditions"].items():
            adapt_ids = {r["recording_id"] for r in cond_data["adaptation_recordings"]}
            eval_ids = {r["recording_id"] for r in cond_data["evaluation_recordings"]}
            overlap = adapt_ids.intersection(eval_ids)
            assert len(overlap) == 0, f"Condition {cond} has overlapping recordings: {overlap}"

    def test_recording_number_assignment(self, manifest):
        for cond, cond_data in manifest["conditions"].items():
            for r in cond_data["adaptation_recordings"]:
                assert 1 <= r["recording_number"] <= 10
            for r in cond_data["evaluation_recordings"]:
                assert 11 <= r["recording_number"] <= 20


class TestPhase17CoralOperatorMath:
    """Verify mathematical properties of fitted CORAL operator."""

    def test_operator_signature_strictly_label_free(self):
        import inspect
        sig_fit = inspect.signature(fit_coral_operator)
        sig_apply = inspect.signature(apply_coral_projection)
        assert "y_target" not in sig_fit.parameters
        assert "labels" not in sig_fit.parameters
        assert "y" not in sig_apply.parameters

    def test_coral_covariance_recoloring(self):
        np.random.seed(42)
        n_s, n_t, d = 600, 800, 16

        # Source features
        L_s = np.random.randn(d, d)
        C_s = L_s @ L_s.T
        X_s = np.random.multivariate_normal(np.ones(d) * 2.0, C_s, size=n_s)

        # Target features
        L_t = np.random.randn(d, d)
        C_t = L_t @ L_t.T
        X_t = np.random.multivariate_normal(np.ones(d) * -3.0, C_t, size=n_t)

        A_coral, mu_s, mu_t = fit_coral_operator(X_s, X_t, regularization=1e-4)
        X_t_aligned = apply_coral_projection(X_t, A_coral, mu_s, mu_t)

        # Mean matching
        np.testing.assert_allclose(np.mean(X_t_aligned, axis=0), mu_s, atol=1e-2)

        # Covariance matching
        C_s_sample = np.cov(X_s, rowvar=False)
        C_aligned_sample = np.cov(X_t_aligned, rowvar=False)
        orig_cov_diff = np.linalg.norm(np.cov(X_t, rowvar=False) - C_s_sample)
        aligned_cov_diff = np.linalg.norm(C_aligned_sample - C_s_sample)
        assert aligned_cov_diff < orig_cov_diff / 3.0


class TestPhase17RecordingAggregation:
    """Verify recording-level voting and probability aggregation logic."""

    def test_aggregation_majority_vote_and_agreement(self):
        rec_ids = ["rec1"] * 5 + ["rec2"] * 5
        y_true = np.array([0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
        # rec1: 4 Normal, 1 Ball -> vote = Normal (agree = 80%)
        # rec2: 5 Inner -> vote = Inner (agree = 100%)
        y_pred = np.array([0, 0, 0, 0, 1, 2, 2, 2, 2, 2])
        probs = np.zeros((10, 4))
        for i, p in enumerate(y_pred):
            probs[i, p] = 0.9
            probs[i, (p + 1) % 4] = 0.1

        res = aggregate_recording_predictions(rec_ids, y_true, y_pred, probs)

        assert res["total_recordings"] == 2
        assert res["recording_accuracy"] == 1.0
        assert res["recording_balanced_accuracy"] == 1.0
        assert res["mean_window_consistency_pct"] == 90.0
        assert len(res["per_recording_details"]) == 2
        assert res["per_recording_details"][0]["window_agreement_pct"] == 80.0
        assert res["per_recording_details"][1]["window_agreement_pct"] == 100.0


class TestPhase17OutputArtifacts:
    """Verify validation_results.json and generalization_summary.json schema."""

    def test_validation_results_reproducibility_audit(self):
        val_path = OUTPUT_DIR / "validation_results.json"
        assert val_path.exists(), f"Missing validation results: {val_path}"
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        assert "phase16_reproducibility_audit" in val_data
        audit = val_data["phase16_reproducibility_audit"]["audit_checks"]
        assert audit["baseline_matches_exactly"] is True
        assert audit["coral_matches_exactly"] is True
        assert audit["source_data_leakage"] == "NONE. Exactly 2,928 CWRU development windows used; 0 holdout recordings."
        assert audit["target_label_leakage"] == "NONE. Target labels never passed into fit_coral_operator or apply_coral_projection."

    def test_generalization_summary_conditions(self):
        gen_path = OUTPUT_DIR / "generalization_summary.json"
        assert gen_path.exists(), f"Missing generalization summary: {gen_path}"
        with open(gen_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        assert "per_condition_comparison" in gen_data
        rows = gen_data["per_condition_comparison"]
        assert len(rows) == 4
        conditions = [r["condition"] for r in rows]
        for c in ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]:
            assert c in conditions

        # Disjoint CORAL should consistently outperform baseline across conditions
        for r in rows:
            assert r["disjoint_coral_win_bal_acc"] > r["baseline_win_bal_acc"]
            assert r["disjoint_coral_rec_bal_acc"] >= r["disjoint_coral_win_bal_acc"]

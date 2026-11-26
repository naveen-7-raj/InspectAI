"""Tests for Phase 16: Order Tracking & Unsupervised Domain Adaptation Investigation.

Verifies:
1. Tachometer audit logic and non-simulation constraints.
2. Unsupervised CORAL covariance alignment mathematical correctness and shape invariance.
3. Strictly label-free adaptation guarantees (no target label leakage).
4. Nominal order-scaling interpolation properties.
5. Schema and integrity of Phase 16 experiment artifacts.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.run_phase16_experiments import (
    audit_order_tracking_feasibility,
    compute_coral_projection,
    evaluate_predictions,
    extract_fused_features,
    REPO_ROOT,
    OUTPUT_DIR,
)


class TestPhase16TachometerAudit:
    """Verify tachometer channel analysis and physical constraints."""

    def test_audit_identifies_cwru_lack_of_tachometer(self):
        audit = audit_order_tracking_feasibility()
        assert "cwru_speed_metadata_audit" in audit
        cwru_meta = audit["cwru_speed_metadata_audit"]
        assert cwru_meta["has_continuous_tachometer"] is False
        assert cwru_meta["has_optical_encoder_pulses"] is False
        assert cwru_meta["has_keyphasor"] is False

    def test_audit_cot_infeasibility_rationale(self):
        audit = audit_order_tracking_feasibility()
        feas = audit["feasibility_assessment"]
        assert feas["dynamic_computed_order_tracking_cot_feasible"] is False
        assert "Simulating synthetic pulses is strictly prohibited" in feas["cot_infeasibility_rationale"]
        assert feas["nominal_order_scaling_feasible"] is True

    def test_paderborn_speed_channel_characteristics(self):
        audit = audit_order_tracking_feasibility()
        pu_speed = audit["paderborn_speed_channel_audit"]
        assert pu_speed["channel_name"] == "speed"
        assert pu_speed["sample_rate_hz"] == 4000.0
        assert 1490.0 < pu_speed["mean_rpm"] < 1510.0


class TestPhase16CoralAlgorithm:
    """Verify mathematical properties of Unsupervised CORAL alignment."""

    def test_coral_projection_shape_and_mean_alignment(self):
        np.random.seed(42)
        n_source, n_target, d = 500, 800, 32
        
        # Source distribution: mean=2.0, cov=diag(1.0)
        X_source = np.random.randn(n_source, d) + 2.0
        # Target distribution: mean=-1.5, cov stretched by 3.0
        X_target = np.random.randn(n_target, d) * 3.0 - 1.5

        X_aligned, mu_s, mu_t = compute_coral_projection(X_source, X_target, regularization=1e-3)

        assert X_aligned.shape == X_target.shape
        # Target mean after alignment should match source mean
        np.testing.assert_allclose(np.mean(X_aligned, axis=0), mu_s, atol=1e-2)

    def test_coral_covariance_matching(self):
        np.random.seed(123)
        n_s, n_t, d = 1000, 1000, 8

        # Generate correlated source
        A_s = np.random.randn(d, d)
        C_s_true = A_s @ A_s.T
        X_source = np.random.multivariate_normal(np.zeros(d), C_s_true, size=n_s)

        # Generate different correlated target
        A_t = np.random.randn(d, d)
        C_t_true = A_t @ A_t.T
        X_target = np.random.multivariate_normal(np.ones(d) * 5.0, C_t_true, size=n_t)

        X_aligned, mu_s, _ = compute_coral_projection(X_source, X_target, regularization=1e-4)

        # Calculate sample covariances
        C_s_sample = np.cov(X_source, rowvar=False)
        C_aligned_sample = np.cov(X_aligned, rowvar=False)

        # Covariance Frobenius distance should be dramatically reduced
        orig_dist = np.linalg.norm(np.cov(X_target, rowvar=False) - C_s_sample)
        aligned_dist = np.linalg.norm(C_aligned_sample - C_s_sample)
        assert aligned_dist < orig_dist / 3.0

    def test_coral_strictly_label_free_signature(self):
        """Verify that compute_coral_projection signature accepts NO target labels."""
        import inspect
        sig = inspect.signature(compute_coral_projection)
        param_names = list(sig.parameters.keys())
        assert "y_target" not in param_names
        assert "labels" not in param_names
        assert "y_source" not in param_names
        assert param_names == ["X_source", "X_target", "regularization"]


class TestPhase16EvaluationMetrics:
    """Verify evaluation metric calculations and confusion matrix structure."""

    def test_evaluation_metric_dictionary(self):
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 0, 3])  # 7/8 correct
        res = evaluate_predictions(y_true, y_pred, "TestExp")

        assert res["total_windows"] == 8
        assert res["accuracy"] == 7 / 8
        assert len(res["confusion_matrix"]["matrix"]) == 4
        assert len(res["confusion_matrix"]["labels"]) == 4
        assert "Normal" in res["per_class_metrics"]


class TestPhase16ArtifactsSchema:
    """Verify schema and validity of Phase 16 output JSON files."""

    def test_feasibility_audit_json_exists_and_valid(self):
        audit_path = OUTPUT_DIR / "feasibility_audit.json"
        assert audit_path.exists(), f"Missing audit file: {audit_path}"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "paderborn_speed_channel_audit" in data
        assert "cwru_speed_metadata_audit" in data
        assert "feasibility_assessment" in data
        assert data["feasibility_assessment"]["dynamic_computed_order_tracking_cot_feasible"] is False

    def test_experiment_results_json_exists_and_valid(self):
        results_path = OUTPUT_DIR / "experiment_results.json"
        assert results_path.exists(), f"Missing results file: {results_path}"
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "experiments" in data
        assert "comparative_summary" in data

        exps = data["experiments"]
        assert "experiment_a_frozen_baseline" in exps
        assert "experiment_b_order_scaling" in exps
        assert "experiment_c_unsupervised_coral" in exps
        assert "experiment_d_combined_order_coral" in exps

        # Check Experiment A matches Phase 14 baseline
        exp_a = exps["experiment_a_frozen_baseline"]
        assert exp_a["total_windows"] == 7211
        assert abs(exp_a["accuracy"] - 0.0829) < 0.001

        # Check Experiment C CORAL gains
        exp_c = exps["experiment_c_unsupervised_coral"]
        assert exp_c["total_windows"] == 7211
        assert exp_c["accuracy"] > 0.50
        assert exp_c["per_class_metrics"]["Normal"]["recall"] > 0.70
        assert exp_c["per_class_metrics"]["Inner Race Fault"]["recall"] > 0.80

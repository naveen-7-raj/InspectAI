"""Phase 20: Lightweight Release Validation Tests.

Validates:
1. Scientific integrity audit results (recording split disjointness, label isolation, threshold provenance).
2. End-to-end integration contracts (REST, fallback, Paderborn adaptation, fault suppression guardrail, rollback).
3. Performance benchmarks (latency SLAs, throughput headroom, memory footprints).
4. Model weights immutability check.
"""

from hashlib import sha256
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN_MODEL_SHA256 = "a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e"


class TestScientificIntegrity:
    """Verify Phase 20 scientific integrity audit findings."""

    def test_audit_artifact_exists_and_sound(self):
        audit_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "scientific_integrity_audit.json"
        assert audit_path.is_file(), "Scientific integrity audit artifact missing"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("overall_integrity_status") == "VERIFIED_SOUND"

    def test_recording_split_strictly_disjoint(self):
        audit_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "scientific_integrity_audit.json"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        split = data["recording_split_integrity"]
        assert split["strictly_disjoint"] is True
        assert split["recording_overlap_count"] == 0
        assert split["total_adaptation_recordings"] == 160
        assert split["total_evaluation_recordings"] == 160

    def test_coral_label_isolation_enforced(self):
        audit_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "scientific_integrity_audit.json"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["coral_label_isolation"]["verified"] is True

    def test_thresholds_free_of_label_leakage(self):
        audit_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "scientific_integrity_audit.json"
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        thresholds = data["threshold_selection_audit"]
        for name, entry in thresholds.items():
            assert entry["evaluated_on_test_labels"] is False, f"Threshold {name} used test labels!"


class TestEndToEndValidation:
    """Verify application smoke and end-to-end integration results."""

    def test_e2e_artifact_exists_and_passed(self):
        e2e_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "end_to_end_validation.json"
        assert e2e_path.is_file(), "End-to-end validation artifact missing"
        with open(e2e_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("all_checks_passed") is True

    def test_smoke_endpoints_healthy(self):
        e2e_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "end_to_end_validation.json"
        with open(e2e_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        smoke = data["smoke_tests"]
        assert smoke["root_health_status"] == 200
        assert smoke["api_health_status"] == 200
        assert smoke["metrics_status"] == 200
        assert smoke["metrics_format_valid"] is True

    def test_cwru_and_fallback_contracts(self):
        e2e_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "end_to_end_validation.json"
        with open(e2e_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        e2e = data["end_to_end_tests"]
        assert e2e["baseline_cwru_prediction"]["status_code"] == 200
        assert e2e["baseline_cwru_prediction"]["predicted_class"] == "Normal"
        assert e2e["adaptive_uncalibrated_fallback"]["fallback_triggered"] is True
        assert e2e["adaptive_uncalibrated_fallback"]["is_adapted"] is False

    def test_real_paderborn_adaptation_success(self):
        e2e_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "end_to_end_validation.json"
        with open(e2e_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pu = data["end_to_end_tests"]["real_paderborn_adaptive_prediction"]
        assert pu.get("adapter_fitted") is True
        assert pu.get("is_adapted") is True
        assert pu.get("condition_number") is not None
        assert pu.get("condition_number") < 1e7

    def test_fault_suspect_suppression_and_rollback(self):
        e2e_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "end_to_end_validation.json"
        with open(e2e_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        e2e = data["end_to_end_tests"]
        guardrail = e2e["fault_suspect_suppression_guardrail"]
        assert guardrail["suppression_verified"] is True
        assert guardrail["recalibrations_during_fault"] == 0

        rollback = e2e["drift_and_rollback_validation"]
        assert rollback["verified"] is True
        assert rollback["rollback_executed"] is True
        assert rollback["state_after_rollback"] == "STABLE"


class TestPerformanceBenchmark:
    """Verify performance metrics meet real-time industrial SLAs."""

    def test_performance_artifact_valid(self):
        perf_path = PROJECT_ROOT / "outputs" / "experiments" / "phase20" / "performance_benchmark.json"
        assert perf_path.is_file(), "Performance benchmark artifact missing"
        with open(perf_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Headroom SLA: at least 4x real-time rate (23.4 windows/sec)
        assert data["throughput_and_stability"]["headroom_factor"] >= 4.0
        # Latency SLA: Mean latency under 50ms per window
        assert data["latency_profile_ms"]["baseline"]["mean_ms"] < 50.0
        assert data["latency_profile_ms"]["adaptive"]["mean_ms"] < 50.0
        # Memory SLA: Memory consumption under 2000 MB
        assert data["system_resources"]["process_rss_memory_after_mb"] < 2000.0


class TestModelCheckpointIntegrity:
    """Verify production checkpoint cryptographic immutability."""

    def test_final_model_checkpoint_sha256(self):
        ckpt_path = PROJECT_ROOT / "models" / "final" / "inspectai_final_model.pth"
        assert ckpt_path.is_file(), "Production model checkpoint not found"
        computed_sha = sha256(ckpt_path.read_bytes()).hexdigest().lower()
        assert computed_sha == FROZEN_MODEL_SHA256, (
            f"MODEL CHECKPOINT TAMPERED! Expected {FROZEN_MODEL_SHA256}, got {computed_sha}"
        )

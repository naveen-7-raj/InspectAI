"""Unit and integration tests for Phase 19 System Hardening & Deployment Readiness.

Verifies:
1. Dockerfile & docker-compose configuration hardening (non-root user, CPU wheel, healthchecks).
2. Prometheus metrics exposition and telemetry tracking (/metrics endpoint).
3. Safe drift detection and shadow recalibration (distinguishing operating shift from fault).
4. Strict safety guardrail: NEVER adapt away a potential fault indication.
5. Cooldown enforcement, rollback guarantees, and frozen model immutability.
"""

from hashlib import sha256
from pathlib import Path
import yaml
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.dependencies import AppState, load_model_artifacts
from backend.main import app
from src.inference.domain_adapter import AdapterStatus, DomainAdapter
from src.inference.drift_monitor import DriftAction, DriftMonitor, DriftState
from src.monitoring.metrics import (
    ADAPTER_CONDITION_NUMBER,
    PREDICTIONS_TOTAL,
    get_metrics_exposition,
)

FROZEN_MODEL_SHA256 = "a612eb9ebcc758dcd078f975414eec2c9409904b30de5fb24588c19990e7542e"


class TestDockerConfigurationHardening:
    """Task 1: Verify production hardening of Docker assets."""

    def test_dockerfile_security_and_cpu_optimization(self):
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        assert "python:3.11-slim" in dockerfile
        assert "useradd" in dockerfile and "USER appuser" in dockerfile
        assert "https://download.pytorch.org/whl/cpu" in dockerfile
        assert "HEALTHCHECK" in dockerfile
        assert "EXPOSE 8000 8501" in dockerfile

    def test_docker_compose_service_contracts(self):
        compose_path = Path("docker-compose.yml")
        assert compose_path.is_file()
        with open(compose_path, "r", encoding="utf-8") as f:
            compose = yaml.safe_load(f)

        services = compose.get("services", {})
        assert "backend" in services
        assert "frontend" in services

        # Backend checks
        backend = services["backend"]
        assert backend["restart"] == "unless-stopped"
        assert "healthcheck" in backend
        assert "limits" in backend["deploy"]["resources"]
        assert any(":ro" in v for v in backend["volumes"])

        # Frontend checks
        frontend = services["frontend"]
        assert frontend["restart"] == "unless-stopped"
        assert "healthcheck" in frontend
        assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"


class TestPrometheusObservability:
    """Task 2: Verify Prometheus metrics endpoint and exposition format."""

    @pytest.fixture
    def test_client(self, monkeypatch):
        monkeypatch.setenv("INSPECTAI_USE_FINAL_MODEL", "1")
        state = AppState()
        load_model_artifacts(state)
        app.state.app_state = state
        with TestClient(app) as client:
            yield client
        # Clean up
        monkeypatch.delenv("INSPECTAI_USE_FINAL_MODEL", raising=False)
        default_state = AppState()
        try:
            load_model_artifacts(default_state)
        except Exception:
            pass
        app.state.app_state = default_state

    def test_metrics_endpoint_exposition(self, test_client):
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        body = resp.text

        # Verify key metric families
        assert "inspectai_predictions_total" in body
        assert "inspectai_inference_latency_seconds" in body
        assert "inspectai_calibration_total" in body
        assert "inspectai_adapter_condition_number" in body
        assert "inspectai_domain_drift_frobenius" in body
        assert "inspectai_active_websockets" in body

    def test_predict_endpoint_updates_metrics(self, test_client):
        signal = [0.01, -0.02, 0.03, -0.01] * 512
        resp = test_client.post(
            "/api/v1/predict",
            json={"signal": signal, "sampling_rate_hz": 12000, "explain": False, "inference_mode": "baseline"},
        )
        assert resp.status_code == 200

        # Scrape metrics again to verify counter increment
        metrics_resp = test_client.get("/metrics")
        assert 'inspectai_predictions_total{inference_mode="baseline"' in metrics_resp.text


class TestSafeDriftDetectionAndRecalibration:
    """Task 3: Verify safe drift detection, fault-suspect suppression, and rollback."""

    @pytest.fixture
    def calibrated_monitor(self):
        rng = np.random.RandomState(42)
        d = 576
        source_features = rng.randn(80, d) * 1.5
        target_adapter = DomainAdapter(min_calibration_samples=50)
        target_adapter.set_source_statistics(source_features)
        for feat in rng.randn(60, d):
            target_adapter.collect_calibration_sample(feat)
        target_adapter.fit()

        monitor = DriftMonitor(
            buffer_size=100,
            drift_frobenius_threshold=0.40,
            fault_prob_threshold=0.35,
            kurtosis_threshold=4.5,
            cooldown_windows=30,
            max_condition_number=1e7,
            min_recalibration_samples=50,
        )
        monitor.attach_adapter(target_adapter)
        return monitor

    def test_stationary_telemetry_remains_stable(self, calibrated_monitor):
        rng = np.random.RandomState(10)
        for _ in range(55):
            f = rng.randn(576)
            w = rng.randn(2048) * 0.05
            p = {"Normal": 0.95, "Ball Fault": 0.02, "Inner Race Fault": 0.02, "Outer Race Fault": 0.01}
            tel = calibrated_monitor.update(f, w, p)
        assert calibrated_monitor.state == DriftState.STABLE
        assert tel["action"] == DriftAction.NONE.value
        assert calibrated_monitor.recalibration_count == 0

    def test_operating_condition_shift_triggers_shadow_recalibration(self, calibrated_monitor):
        rng = np.random.RandomState(20)
        d = 576
        shift = np.eye(d) * 2.2 + rng.randn(d, d) * 0.08
        actions = []
        for _ in range(65):
            f = rng.randn(d) @ shift
            w = rng.randn(2048) * 0.08
            p = {"Normal": 0.90, "Ball Fault": 0.03, "Inner Race Fault": 0.04, "Outer Race Fault": 0.03}
            tel = calibrated_monitor.update(f, w, p)
            actions.append(tel["action"])

        assert DriftAction.SHADOW_PROMOTED.value in actions
        assert calibrated_monitor.recalibration_count >= 1

    def test_strict_safety_guardrail_fault_suppresses_adaptation(self, calibrated_monitor):
        """CRITICAL: Ensure that when bearing fault emerges, adaptation is blocked."""
        rng = np.random.RandomState(30)
        d = 576
        actions = []
        initial_recalibs = calibrated_monitor.recalibration_count

        for _ in range(60):
            # Impulsive defect signal
            w = rng.randn(2048) * 0.05
            w[rng.choice(2048, size=20, replace=False)] += 2.0  # Spikes: kurtosis > 5.0
            f = rng.randn(d) * 3.0 + 1.5
            p = {"Normal": 0.10, "Ball Fault": 0.10, "Inner Race Fault": 0.75, "Outer Race Fault": 0.05}
            tel = calibrated_monitor.update(f, w, p)
            actions.append(tel["action"])

        # Safety assertions
        assert DriftAction.SUPPRESSED_FAULT_SUSPECT.value in actions
        assert calibrated_monitor.recalibration_count == initial_recalibs
        assert tel["is_fault_suspect"] is True

    def test_cooldown_enforcement(self, calibrated_monitor):
        rng = np.random.RandomState(40)
        # Buffer 60 normal windows so feature buffer has >= min_recalibration_samples (50)
        for _ in range(60):
            calibrated_monitor.update(
                rng.randn(576) * 1.0,
                rng.randn(2048) * 0.05,
                {"Normal": 0.98, "Ball Fault": 0.01, "Inner Race Fault": 0.01, "Outer Race Fault": 0.0},
            )

        calibrated_monitor.windows_since_recalibration = 10  # Under cooldown limit (30)
        actions = []
        for _ in range(20):
            f = rng.randn(576) * 4.0
            w = rng.randn(2048) * 0.05
            p = {"Normal": 0.95, "Ball Fault": 0.02, "Inner Race Fault": 0.02, "Outer Race Fault": 0.01}
            tel = calibrated_monitor.update(f, w, p)
            actions.append(tel["action"])

        assert DriftAction.COOLDOWN_ACTIVE.value in actions

    def test_adapter_rollback_mechanism(self, calibrated_monitor):
        rng = np.random.RandomState(42)
        # Feed 60 windows so feature buffer has >= min_recalibration_samples (50)
        for _ in range(60):
            calibrated_monitor.update(
                rng.randn(576) * 1.5,
                rng.randn(2048) * 0.05,
                {"Normal": 0.95, "Ball Fault": 0.02, "Inner Race Fault": 0.02, "Outer Race Fault": 0.01},
            )
        # Promote an adapter first
        action = calibrated_monitor._execute_shadow_recalibration()
        assert action == DriftAction.SHADOW_PROMOTED
        assert calibrated_monitor.previous_adapter is not None

        # Execute rollback
        success = calibrated_monitor.rollback(reason="test_instability")
        assert success is True
        assert calibrated_monitor.rollback_count == 1
        assert calibrated_monitor.previous_adapter is None


class TestModelImmutability:
    """Verify cryptographic integrity of production model checkpoint."""

    def test_model_weights_sha256_unaltered(self):
        ckpt_path = Path("models/final/inspectai_final_model.pth")
        assert ckpt_path.is_file()
        current_sha256 = sha256(ckpt_path.read_bytes()).hexdigest().lower()
        assert current_sha256 == FROZEN_MODEL_SHA256

"""InspectAI Safe Drift Detection & Shadow Recalibration Engine (Phase 19).

Monitors feature statistics during continuous streaming to detect operational domain drift
while enforcing safety guardrails:
1. Distinguishes ordinary operating-condition changes (speed/load shifts) from fault emergence.
2. STRICT SAFETY RULE: Never automatically adapt away a potential fault indication.
   If signal impulsiveness (kurtosis) or model fault posterior is elevated, recalibration is SUPPRESSED.
3. Shadow Recalibration: Calibrates a candidate adapter in shadow mode and verifies numerical
   stability (sample count, finite condition number) before activation.
4. Cooldown Enforcement: Enforces minimum sample intervals between recalibrations to prevent fluttering.
5. Rollback & Baseline Fallback: Retains the previous valid adapter in memory for zero-downtime
   rollback, falling back to frozen baseline inference if stability degrades.
"""

from collections import deque
from enum import Enum
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import kurtosis

from src.inference.domain_adapter import AdapterStatus, DomainAdapter
from src.monitoring.metrics import record_recalibration, record_rollback

logger = logging.getLogger("InspectAI.DriftMonitor")


class DriftState(str, Enum):
    """Operational drift status."""
    STABLE = "STABLE"
    OPERATING_SHIFT_DETECTED = "OPERATING_SHIFT_DETECTED"
    FAULT_SUSPECT = "FAULT_SUSPECT"
    COOLDOWN = "COOLDOWN"
    SHADOW_CALIBRATING = "SHADOW_CALIBRATING"
    FALLBACK = "FALLBACK"


class DriftAction(str, Enum):
    """Action taken by drift monitor on each inspection cycle."""
    NONE = "NONE"
    SUPPRESSED_FAULT_SUSPECT = "SUPPRESSED_FAULT_SUSPECT"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    SHADOW_PROMOTED = "SHADOW_PROMOTED"
    SHADOW_REJECTED_INSTABILITY = "SHADOW_REJECTED_INSTABILITY"
    ROLLBACK_EXECUTED = "ROLLBACK_EXECUTED"
    FALLBACK_BASELINE = "FALLBACK_BASELINE"


class DriftMonitor:
    """Safe, non-intrusive drift monitor and shadow recalibration coordinator."""

    def __init__(
        self,
        buffer_size: int = 100,
        drift_frobenius_threshold: float = 0.40,
        fault_prob_threshold: float = 0.35,
        kurtosis_threshold: float = 4.5,
        cooldown_windows: int = 50,
        max_condition_number: float = 1e7,
        min_recalibration_samples: int = 50,
    ) -> None:
        """Initialize drift monitor.

        Args:
            buffer_size: Number of rolling feature vectors retained for drift detection.
            drift_frobenius_threshold: Relative Frobenius distance threshold triggering recalibration.
            fault_prob_threshold: Cumulative non-normal class probability above which fault is suspected.
            kurtosis_threshold: Vibration signal kurtosis above which impact defect is suspected.
            cooldown_windows: Minimum window evaluations between consecutive recalibrations.
            max_condition_number: Maximum allowed condition number for shadow adapter promotion.
            min_recalibration_samples: Minimum samples required to attempt shadow recalibration.
        """
        self.buffer_size = buffer_size
        self.drift_frobenius_threshold = drift_frobenius_threshold
        self.fault_prob_threshold = fault_prob_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.cooldown_windows = cooldown_windows
        self.max_condition_number = max_condition_number
        self.min_recalibration_samples = min_recalibration_samples

        # Rolling buffers
        self._feature_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self._kurtosis_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._fault_prob_buffer: Deque[float] = deque(maxlen=buffer_size)

        # State tracking
        self.state = DriftState.STABLE
        self.active_adapter: Optional[DomainAdapter] = None
        self.previous_adapter: Optional[DomainAdapter] = None
        self.reference_covariance: Optional[np.ndarray] = None
        self.windows_since_recalibration: int = cooldown_windows  # Start eligible
        self.total_windows_observed: int = 0
        self.last_drift_distance: float = 0.0
        self.recalibration_count: int = 0
        self.rollback_count: int = 0

    def attach_adapter(self, adapter: DomainAdapter) -> None:
        """Attach initial calibrated domain adapter as active."""
        self.active_adapter = adapter
        if adapter.is_fitted and adapter.target_covariance is not None:
            self.reference_covariance = adapter.target_covariance.copy()
            self.state = DriftState.STABLE

    def update(
        self,
        features: np.ndarray,
        vibration_window: np.ndarray,
        class_probabilities: Dict[str, float],
    ) -> Dict[str, Any]:
        """Ingest new window telemetry and evaluate drift & safety conditions.

        Args:
            features: 1D latent feature array (e.g. 576-dim).
            vibration_window: Raw vibration signal window.
            class_probabilities: Softmax class probabilities from predictor.

        Returns:
            Telemetry dict with drift metrics, state, and action taken.
        """
        self.total_windows_observed += 1
        self.windows_since_recalibration += 1

        # 1. Update rolling buffers
        self._feature_buffer.append(features.flatten().astype(np.float64))

        # Compute signal kurtosis (unbiased Fisher kurtosis + 3 = Pearson kurtosis)
        try:
            k_val = float(kurtosis(vibration_window, fisher=False))
        except Exception:
            k_val = 3.0
        self._kurtosis_buffer.append(k_val)

        # Fault probability (sum of Ball, Inner Race, Outer Race)
        non_normal_prob = 1.0 - float(class_probabilities.get("Normal", 1.0))
        self._fault_prob_buffer.append(non_normal_prob)

        # 2. Safety Assessment: Check for potential fault emergence
        recent_kurtosis = float(np.mean(list(self._kurtosis_buffer)[-20:])) if len(self._kurtosis_buffer) >= 5 else 3.0
        recent_fault_prob = float(np.mean(list(self._fault_prob_buffer)[-20:])) if len(self._fault_prob_buffer) >= 5 else 0.0

        is_fault_suspect = (recent_fault_prob >= self.fault_prob_threshold) or (recent_kurtosis >= self.kurtosis_threshold)

        # 3. Compute domain drift if reference covariance exists
        drift_distance = 0.0
        drift_detected = False

        if len(self._feature_buffer) >= self.min_recalibration_samples and self.reference_covariance is not None:
            # Estimate rolling feature variance (diagonal of covariance) to avoid O(d^2/N) off-diagonal noise in d=576
            X_rolling = np.array(self._feature_buffer)
            var_rolling = np.var(X_rolling, axis=0) + 1e-4
            var_ref = np.diag(self.reference_covariance) if self.reference_covariance.ndim == 2 else self.reference_covariance
            var_ref = np.maximum(var_ref, 1e-4)

            ref_norm = np.linalg.norm(var_ref)
            if ref_norm > 1e-9:
                diff_norm = np.linalg.norm(var_rolling - var_ref)
                drift_distance = float(diff_norm / ref_norm)
                self.last_drift_distance = drift_distance

            if drift_distance >= self.drift_frobenius_threshold:
                drift_detected = True

        # 4. Decide State & Action with Safety Guardrail
        action = DriftAction.NONE

        if is_fault_suspect:
            self.state = DriftState.FAULT_SUSPECT
            if drift_detected:
                # CRITICAL: Suppress adaptation during fault conditions
                action = DriftAction.SUPPRESSED_FAULT_SUSPECT
                logger.warning(
                    f"[SAFETY ALERT] Drift detected ({drift_distance:.2f} >= {self.drift_frobenius_threshold:.2f}) "
                    f"but RECALIBRATION SUPPRESSED due to fault suspicion (P(Fault)={recent_fault_prob:.2f}, "
                    f"Kurtosis={recent_kurtosis:.2f}). Never adapt away fault signatures."
                )
        elif drift_detected:
            self.state = DriftState.OPERATING_SHIFT_DETECTED
            # Check cooldown
            if self.windows_since_recalibration < self.cooldown_windows:
                action = DriftAction.COOLDOWN_ACTIVE
                self.state = DriftState.COOLDOWN
                logger.info(
                    f"Drift detected ({drift_distance:.2f}) but cooldown active "
                    f"({self.windows_since_recalibration}/{self.cooldown_windows} windows). Postponing recalibration."
                )
            else:
                # Trigger Shadow Recalibration
                action = self._execute_shadow_recalibration()
        else:
            self.state = DriftState.STABLE

        return {
            "state": self.state.value,
            "action": action.value,
            "drift_distance": round(drift_distance, 4),
            "drift_threshold": self.drift_frobenius_threshold,
            "recent_fault_prob": round(recent_fault_prob, 4),
            "recent_kurtosis": round(recent_kurtosis, 2),
            "is_fault_suspect": is_fault_suspect,
            "windows_since_recalibration": self.windows_since_recalibration,
            "buffered_windows": len(self._feature_buffer),
        }

    def _execute_shadow_recalibration(self) -> DriftAction:
        """Execute shadow recalibration on rolling feature buffer with strict stability gating."""
        if self.active_adapter is None or not self.active_adapter.is_fitted:
            return DriftAction.FALLBACK_BASELINE

        self.state = DriftState.SHADOW_CALIBRATING
        logger.info("Initiating shadow adapter recalibration on rolling buffer...")

        try:
            # 1. Instantiate candidate shadow adapter copying source statistics
            shadow_adapter = DomainAdapter(
                source_mean=self.active_adapter.source_mean.copy() if self.active_adapter.source_mean is not None else None,
                source_cov_sqrt=self.active_adapter.source_cov_sqrt.copy() if self.active_adapter.source_cov_sqrt is not None else None,
                min_calibration_samples=self.min_recalibration_samples,
                regularization=self.active_adapter.regularization,
                max_condition_number=self.max_condition_number,
            )

            # 2. Buffer rolling features
            for feat_vec in self._feature_buffer:
                shadow_adapter.collect_calibration_sample(feat_vec)

            # 3. Fit shadow adapter
            fit_success = shadow_adapter.fit()
            if not fit_success:
                record_recalibration(trigger="operating_drift", outcome="fit_failed")
                return DriftAction.SHADOW_REJECTED_INSTABILITY

            # 4. Numerical Stability Gate
            cond_num = shadow_adapter.condition_number
            if cond_num > self.max_condition_number or not np.isfinite(cond_num):
                logger.warning(
                    f"Shadow adapter rejected: condition number ({cond_num:.1f}) exceeds "
                    f"safety threshold ({self.max_condition_number:.1f})."
                )
                record_recalibration(trigger="operating_drift", outcome="rejected_ill_conditioned")
                return DriftAction.SHADOW_REJECTED_INSTABILITY

            # 5. Promotion & Backup
            self.previous_adapter = self.active_adapter
            self.active_adapter = shadow_adapter
            self.reference_covariance = shadow_adapter.target_covariance.copy()
            self.windows_since_recalibration = 0
            self.recalibration_count += 1
            self.state = DriftState.STABLE

            record_recalibration(trigger="operating_drift", outcome="promoted")
            logger.info(
                f"[RECALIBRATION PROMOTED] Shadow adapter successfully activated. "
                f"Samples={len(self._feature_buffer)}, Condition Number={cond_num:.1f}. Active adapter updated."
            )
            return DriftAction.SHADOW_PROMOTED

        except Exception as exc:
            logger.error(f"Shadow recalibration failed with unexpected error: {exc}")
            record_recalibration(trigger="operating_drift", outcome=f"error_{type(exc).__name__}")
            return DriftAction.SHADOW_REJECTED_INSTABILITY

    def rollback(self, reason: str = "instability_detected") -> bool:
        """Roll back active adapter to previously valid adapter instance."""
        if self.previous_adapter is not None and self.previous_adapter.is_fitted:
            logger.warning(f"Executing adapter rollback: reverting to previous valid adapter (reason: {reason}).")
            self.active_adapter = self.previous_adapter
            self.reference_covariance = self.previous_adapter.target_covariance.copy()
            self.previous_adapter = None
            self.state = DriftState.STABLE
            self.rollback_count += 1
            record_rollback(reason=reason)
            return True
        else:
            logger.error("Rollback failed: no valid previous adapter available. Reverting to baseline fallback.")
            if self.active_adapter is not None:
                self.active_adapter.status = AdapterStatus.FALLBACK
            self.state = DriftState.FALLBACK
            record_rollback(reason="fallback_no_previous_adapter")
            return False

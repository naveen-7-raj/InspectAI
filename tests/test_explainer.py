"""Unit tests for 1D Grad-CAM Explainability Engine (Phase 5)."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability.explainer import VibrationExplainer, DEFAULT_CLASS_NAMES
from src.models.cnn_1d import VibrationCNN1D
from src.training.normalizer import VibrationNormalizer
from src.utils.config_parser import get_resolved_paths, load_config


class TestVibrationExplainer:
    """Test suite verifying 1D Grad-CAM implementation, hook mechanics, attribution properties, and test integration."""

    @pytest.fixture
    def config(self):
        return load_config()

    @pytest.fixture
    def resolved_paths(self, config):
        return get_resolved_paths(config)

    @pytest.fixture
    def base_model(self, config):
        torch.manual_seed(42)
        model = VibrationCNN1D.from_config(config)
        model.eval()
        return model

    @pytest.fixture
    def normalizer(self):
        norm = VibrationNormalizer()
        norm.mean = 0.015
        norm.std = 0.25
        norm.is_fitted = True
        return norm

    def test_explainer_initialization(self, base_model, normalizer):
        """Verify proper default target layer resolution, device assignment, and attributes."""
        explainer = VibrationExplainer(
            model=base_model,
            normalizer=normalizer,
            device="cpu",
        )
        assert explainer.target_layer is base_model.conv3
        assert explainer.normalizer is normalizer
        assert explainer.class_names == DEFAULT_CLASS_NAMES
        assert explainer.device == torch.device("cpu")

    def test_custom_target_layer(self, base_model):
        """Verify custom target layer assignment."""
        custom_layer = base_model.block2[0]  # First conv in block2
        explainer = VibrationExplainer(model=base_model, target_layer=custom_layer)
        assert explainer.target_layer is custom_layer

    def test_gradcam_output_shape_and_bounds(self, base_model):
        """Verify that Grad-CAM returns a 1D array of length 2048 with values bounded in [0, 1]."""
        explainer = VibrationExplainer(model=base_model)
        x = np.random.randn(2048).astype(np.float32)

        heatmap, pred_class, expl_class, logits, probs = explainer.compute_gradcam(x)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (2048,)
        assert np.all(np.isfinite(heatmap))
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        assert 0 <= pred_class < 4
        assert expl_class == pred_class
        assert len(logits) == 4
        assert len(probs) == 4
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    def test_gradcam_target_class_selection(self, base_model):
        """Verify that target_class specifies which logit is differentiated."""
        explainer = VibrationExplainer(model=base_model)
        x = np.random.randn(2048).astype(np.float32)

        for target_c in range(4):
            heatmap, pred_c, expl_c, logits, probs = explainer.compute_gradcam(x, target_class=target_c)
            assert expl_c == target_c
            assert heatmap.shape == (2048,)
            assert np.all(np.isfinite(heatmap))
            assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    def test_target_class_out_of_bounds(self, base_model):
        """Verify ValueError when target class index is invalid."""
        explainer = VibrationExplainer(model=base_model)
        x = np.random.randn(2048).astype(np.float32)

        with pytest.raises(ValueError, match="out of bounds"):
            explainer.compute_gradcam(x, target_class=10)

        with pytest.raises(ValueError, match="out of bounds"):
            explainer.compute_gradcam(x, target_class=-1)

    def test_model_weights_remain_strictly_unmodified(self, base_model):
        """Verify that computing Grad-CAM does not mutate any model parameter weights."""
        explainer = VibrationExplainer(model=base_model)
        x = np.random.randn(2048).astype(np.float32)

        # Snapshot weights before
        weights_before = [p.clone().detach() for p in base_model.parameters()]

        # Run several Grad-CAM passes
        for target_c in [None, 0, 1, 2, 3]:
            explainer.compute_gradcam(x, target_class=target_c)

        # Compare weights after
        for p_before, p_after in zip(weights_before, base_model.parameters()):
            assert torch.equal(p_before, p_after), "Model weights were modified during Grad-CAM execution!"

    def test_hook_cleanup_and_gradient_zeroing(self, base_model):
        """Verify hooks are removed and model gradients are zeroed after computation."""
        explainer = VibrationExplainer(model=base_model)
        x = np.random.randn(2048).astype(np.float32)

        explainer.compute_gradcam(x)

        # Forward/backward hook dicts should be empty
        assert len(explainer.target_layer._forward_hooks) == 0
        assert len(explainer.target_layer._backward_hooks) == 0

        # Model parameter gradients should be None or zero
        for p in base_model.parameters():
            if p.grad is not None:
                assert torch.all(p.grad == 0), "Gradients were not zeroed after Grad-CAM pass."

    def test_explain_window_with_normalizer(self, base_model, normalizer):
        """Verify explain_window end-to-end with raw and normalized inputs."""
        explainer = VibrationExplainer(model=base_model, normalizer=normalizer)
        raw_waveform = np.random.randn(2048).astype(np.float32) * 0.25 + 0.015

        res = explainer.explain_window(
            vibration_window=raw_waveform,
            target_class="Ball Fault",
            is_normalized=False,
            sampling_rate_hz=12000,
            importance_threshold=0.5,
        )

        assert res["predicted_class_id"] in [0, 1, 2, 3]
        assert res["target_class_id"] == 1
        assert res["target_class_name"] == "Ball Fault"
        assert len(res["gradcam_heatmap"]) == 2048
        assert len(res["timestamps_sec"]) == 2048
        assert res["timestamps_sec"][-1] == pytest.approx(2047 / 12000, rel=1e-4)
        assert isinstance(res["important_regions"], list)
        assert "uncalibrated" in res["confidence_note"].lower()

    def test_explain_window_string_class_lookup(self, base_model):
        """Verify target_class string matching is robust to casing and formatting."""
        explainer = VibrationExplainer(model=base_model)
        x = np.random.randn(2048).astype(np.float32)

        res1 = explainer.explain_window(x, target_class="inner_race_fault", is_normalized=True)
        assert res1["target_class_id"] == 2
        assert res1["target_class_name"] == "Inner Race Fault"

        res2 = explainer.explain_window(x, target_class="outer race fault", is_normalized=True)
        assert res2["target_class_id"] == 3
        assert res2["target_class_name"] == "Outer Race Fault"

        with pytest.raises(ValueError, match="not found"):
            explainer.explain_window(x, target_class="non_existent_class", is_normalized=True)

    def test_zero_gradient_or_activation_safely_handled(self, config):
        """Verify numerical stability when activations or gradients are completely zero."""
        model = VibrationCNN1D.from_config(config)
        model.eval()
        # Zero out conv3 weights and bias so activations are exactly zero
        model.conv3.weight.data.zero_()
        model.conv3.bias.data.zero_()

        explainer_zero = VibrationExplainer(model=model)
        x = np.random.randn(2048).astype(np.float32)

        # Should safely return zeros without NaN or division by zero error
        heatmap, pred_c, expl_c, logits, probs = explainer_zero.compute_gradcam(x)
        assert heatmap.shape == (2048,)
        assert np.all(heatmap == 0.0)
        assert np.all(np.isfinite(heatmap))

    def test_frozen_final_checkpoint_explainability(self, resolved_paths, config):
        """Verify Grad-CAM on actual frozen InspectAI final model checkpoint."""
        model_path = resolved_paths["model_checkpoint_dir"] / "inspectai_final_cnn.pth"
        norm_path = resolved_paths["model_checkpoint_dir"] / "inspectai_final_normalizer.json"

        if not model_path.is_file() or not norm_path.is_file():
            pytest.skip("Frozen checkpoint or normalizer not available.")

        # Load real frozen model from checkpoint dictionary
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = VibrationCNN1D.from_config(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Load real normalizer
        normalizer = VibrationNormalizer.load(norm_path)

        explainer = VibrationExplainer(model=model, normalizer=normalizer)
        raw_signal = np.sin(np.linspace(0, 50 * np.pi, 2048)).astype(np.float32) * 0.5

        result = explainer.explain_window(raw_signal, is_normalized=False)
        assert result["gradcam_heatmap"].shape == (2048,)
        assert np.all(np.isfinite(result["gradcam_heatmap"]))
        assert result["target_class_name"] in DEFAULT_CLASS_NAMES

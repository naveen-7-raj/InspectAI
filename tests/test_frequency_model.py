"""Unit tests for the dual-branch SpectralFusionCNN architecture.

Tests verify:
1. SpectralFeatureBranch forward pass, output shape, and parameter efficiency.
2. SpectralFusionCNN forward pass with combined (B, 1, 3073) and separate inputs.
3. Parameter count verification (~89.6k parameters).
4. Backward pass and gradient flow through both waveform and spectral branches.
5. CAM target layer access for explainability compatibility.
"""

import pytest
import torch

from src.models.spectral_fusion_cnn import SpectralFeatureBranch, SpectralFusionCNN


class TestSpectralFeatureBranch:
    """Unit tests for the spectral feature extraction branch."""

    def test_output_shape(self):
        """Input of shape (B, 1, 1025) produces embedding (B, 64)."""
        branch = SpectralFeatureBranch(in_bins=1025, embedding_dim=64)
        x = torch.randn(8, 1, 1025)
        out = branch(x)

        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

    def test_parameter_budget(self):
        """Spectral branch should be lightweight (< 15,000 parameters)."""
        branch = SpectralFeatureBranch(in_bins=1025, embedding_dim=64)
        total_params = sum(p.numel() for p in branch.parameters())
        assert total_params < 15000


class TestSpectralFusionCNN:
    """Unit tests for the dual-branch SpectralFusionCNN."""

    def test_combined_input_forward(self):
        """Combined input tensor (B, 1, 3073) passes through waveform and spectral branches."""
        model = SpectralFusionCNN(spectral_type="fft")
        x_combined = torch.randn(4, 1, 3073)
        logits = model(x_combined)

        assert logits.shape == (4, 4)
        assert not torch.isnan(logits).any()

    def test_separate_inputs_forward(self):
        """Separate waveform (B, 1, 2048) and spectrum (B, 1, 1025) produce identical shape."""
        model = SpectralFusionCNN(spectral_type="hilbert")
        x_wave = torch.randn(4, 1, 2048)
        x_spec = torch.randn(4, 1, 1025)
        logits = model(x_wave, x_spec)

        assert logits.shape == (4, 4)
        assert not torch.isnan(logits).any()

    def test_parameter_count(self):
        """Total trainable parameters should be exactly 89,604."""
        model = SpectralFusionCNN()
        n_params = model.count_parameters()
        assert n_params == 89604

    def test_backward_gradient_flow(self):
        """Backpropagating loss through combined input produces non-zero gradients in both branches."""
        model = SpectralFusionCNN()
        x_combined = torch.randn(2, 1, 3073, requires_grad=True)
        target = torch.tensor([1, 2], dtype=torch.long)

        logits = model(x_combined)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        # Check gradients in waveform branch
        assert model.wave_block1[0].weight.grad is not None
        assert torch.norm(model.wave_block1[0].weight.grad) > 0.0

        # Check gradients in spectral branch
        assert model.spectral_branch.conv1.weight.grad is not None
        assert torch.norm(model.spectral_branch.conv1.weight.grad) > 0.0

        # Check gradients in fusion head
        assert model.classifier[1].weight.grad is not None
        assert torch.norm(model.classifier[1].weight.grad) > 0.0

    def test_cam_target_layer(self):
        """Waveform branch provides conv3 as CAM target layer for Grad-CAM."""
        model = SpectralFusionCNN()
        cam_layer = model.get_cam_target_layer()
        assert cam_layer is model.conv3
        assert model.get_target_layer_for_cam() is model.conv3

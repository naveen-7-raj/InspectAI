"""Unit tests for MultiscaleVibrationCNN1D architecture (Phase 10.3)."""

import pytest
import torch
import torch.nn as nn

from src.models.multiscale_cnn_1d import MultiscaleTemporalBlock, MultiscaleVibrationCNN1D


class TestMultiscaleVibrationCNN1D:
    """Test suite verifying multiscale 1D CNN forward pass, backward pass, and branch behaviors."""

    def test_multiscale_block_output_shapes_and_branches(self):
        """Verify parallel branch execution, temporal alignment, and concatenation."""
        block = MultiscaleTemporalBlock(in_channels=1, branch_channels=16, kernel_sizes=(5, 15, 65))
        assert block.out_channels == 48

        x = torch.randn(4, 1, 2048)
        out = block(x)

        assert out.shape == (4, 48, 2048)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        # Check diagnostic branch outputs
        branches = block.forward_branches(x)
        assert len(branches) == 3
        for k in [5, 15, 65]:
            assert f"branch_{k}" in branches
            assert branches[f"branch_{k}"].shape == (4, 16, 2048)

    def test_forward_output_shape_and_logits(self):
        """Verify forward pass on standard batch yields unnormalized (B, 4) logits."""
        model = MultiscaleVibrationCNN1D(num_classes=4, input_channels=1, input_length=2048)
        x = torch.randn(8, 1, 2048)
        logits = model(x)

        assert logits.shape == (8, 4)
        assert logits.dtype == torch.float32
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_backward_pass_gradient_propagation_all_branches(self):
        """Verify gradients flow back through all 3 multiscale branches simultaneously."""
        model = MultiscaleVibrationCNN1D(num_classes=4, input_channels=1, input_length=2048)
        x = torch.randn(4, 1, 2048, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Check gradient in inputs
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradients in each multiscale branch
        block = model.get_multiscale_block()
        for branch, name in [(block.branch1, "branch_5"), (block.branch2, "branch_15"), (block.branch3, "branch_65")]:
            conv = branch[0]
            assert conv.weight.grad is not None, f"No gradient in {name} conv weights!"
            assert not torch.isnan(conv.weight.grad).any()
            assert torch.norm(conv.weight.grad).item() > 0.0

    def test_parameter_count_and_comparison(self):
        """Verify parameter count is within the expected ~81k range."""
        model = MultiscaleVibrationCNN1D()
        param_count = model.count_parameters()

        # Expected: ~81,252 parameters (well within CPU budget)
        assert 70000 <= param_count <= 95000
        assert param_count == 81252

    def test_deterministic_initialization(self):
        """Verify identical seed yields identical initial weights and outputs."""
        torch.manual_seed(1234)
        m1 = MultiscaleVibrationCNN1D()

        torch.manual_seed(1234)
        m2 = MultiscaleVibrationCNN1D()

        x = torch.randn(2, 1, 2048)
        m1.eval()
        m2.eval()

        with torch.no_grad():
            out1 = m1(x)
            out2 = m2(x)

        assert torch.allclose(out1, out2, atol=1e-7)

    def test_invalid_input_shape_rejection(self):
        """Verify dimensional checks reject 2D or wrong channel tensors."""
        model = MultiscaleVibrationCNN1D(input_channels=1)

        # 2D tensor
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            model(torch.randn(4, 2048))

        # Wrong channels (2 instead of 1)
        with pytest.raises(ValueError, match="Expected 1 input channel"):
            model(torch.randn(4, 2, 2048))

    def test_target_layer_for_cam(self):
        """Verify Grad-CAM target layer is conv3 (deep feature representation)."""
        model = MultiscaleVibrationCNN1D()
        target = model.get_target_layer_for_cam()

        assert isinstance(target, nn.Conv1d)
        assert target == model.conv3
        assert target.out_channels == 128


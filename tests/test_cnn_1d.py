"""Unit tests for VibrationCNN1D architecture (Phase 4, Task 4.1)."""

import pytest
import torch
import torch.nn as nn

from src.models.cnn_1d import VibrationCNN1D
from src.utils.config_parser import load_config


class TestVibrationCNN1D:
    """Test suite verifying 1D CNN architecture specification, forward pass, and layer hooks."""

    def test_instantiation_defaults(self):
        """Verify default parameters and submodules."""
        model = VibrationCNN1D()
        assert model.num_classes == 4
        assert model.input_channels == 1
        assert model.input_length == 2048
        assert hasattr(model, "block1")
        assert hasattr(model, "block2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "classifier")

    def test_from_config(self):
        """Verify instantiation from project configuration."""
        config = load_config()
        model = VibrationCNN1D.from_config(config)
        assert isinstance(model, VibrationCNN1D)
        assert model.num_classes == 4
        assert model.input_channels == 1
        assert model.input_length == 2048

    def test_forward_pass_output_shape(self):
        """Verify output tensor shapes for different batch sizes."""
        model = VibrationCNN1D()
        model.eval()

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 1, 2048, dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (batch_size, 4), f"Expected shape ({batch_size}, 4), got {out.shape}"
            assert out.dtype == torch.float32

    def test_forward_pass_raw_logits(self):
        """Verify outputs are raw unnormalized logits (not constrained to [0, 1] probability range)."""
        model = VibrationCNN1D()
        model.eval()
        x = torch.randn(8, 1, 2048, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)

        # Logits should not all be in [0, 1] and should not sum to 1 along dim=1
        sums = logits.sum(dim=1)
        assert not torch.allclose(sums, torch.ones_like(sums)), "Model should return raw logits, not softmax outputs."

    def test_target_layer_for_cam(self):
        """Verify target layer for Grad-CAM explainability is conv3."""
        model = VibrationCNN1D()
        target_layer = model.get_target_layer_for_cam()
        assert target_layer is model.conv3
        assert isinstance(target_layer, nn.Conv1d)
        assert target_layer.out_channels == 128

    def test_parameter_count(self):
        """Verify parameter count is within reasonable CPU budget (~73k parameters)."""
        model = VibrationCNN1D()
        params = model.count_parameters()
        assert params > 50000, f"Parameter count too small: {params}"
        assert params < 200000, f"Parameter count exceeds CPU budget: {params}"

    def test_invalid_input_dimensions(self):
        """Verify improper tensor dimensions raise informative ValueErrors."""
        model = VibrationCNN1D()

        # 2D tensor (batch, seq_len) instead of (batch, channel, seq_len)
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            model(torch.randn(8, 2048))

        # 4D tensor
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            model(torch.randn(8, 1, 1, 2048))

        # Wrong number of input channels (e.g. 2 channels instead of 1)
        with pytest.raises(ValueError, match="Expected 1 input channel"):
            model(torch.randn(8, 2, 2048))

    def test_backward_pass_gradients(self):
        """Verify backward gradient flow through all convolutional blocks and classifier."""
        model = VibrationCNN1D()
        model.train()

        x = torch.randn(4, 1, 2048, requires_grad=True)
        target = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()

        # Check gradients exist on conv layers and classifier
        assert model.block1[0].weight.grad is not None
        assert model.block2[0].weight.grad is not None
        assert model.conv3.weight.grad is not None
        assert model.classifier[1].weight.grad is not None
        assert not torch.isnan(model.conv3.weight.grad).any()


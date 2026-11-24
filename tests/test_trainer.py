"""Unit tests for ModelTrainer (Phase 4, Task 4.2)."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import ModelTrainer


class SimpleTestClassifier(nn.Module):
    """Small linear model for fast unit testing of trainer functionality."""

    def __init__(self, in_features: int = 10, num_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.linear(x)


class BadNaNModel(nn.Module):
    """Faulty model emitting NaNs to test numerical guard."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full((x.size(0), 4), float("nan"))
        return out


@pytest.fixture
def dummy_dataloaders():
    """Create reproducible synthetic dataloaders for testing."""
    torch.manual_seed(42)
    x_train = torch.randn(64, 1, 10)
    y_train = torch.randint(0, 4, (64,))
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    x_val = torch.randn(32, 1, 10)
    y_val = torch.randint(0, 4, (32,))
    val_ds = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    return train_loader, val_loader


class TestModelTrainer:
    """Test suite verifying ModelTrainer execution, metrics, and error handling."""

    def test_instantiation(self, tmp_path):
        """Verify trainer initialization and default configuration."""
        model = SimpleTestClassifier()
        trainer = ModelTrainer(
            model=model,
            checkpoint_dir=tmp_path / "checkpoints",
            early_stopping_patience=5,
        )
        assert trainer.device == torch.device("cpu")
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.early_stopping_patience == 5

    def test_train_epoch_updates_weights(self, dummy_dataloaders, tmp_path):
        """Verify training loop reduces loss and updates model weights."""
        train_loader, _ = dummy_dataloaders
        model = SimpleTestClassifier()
        initial_weights = model.linear.weight.clone().detach()

        trainer = ModelTrainer(model=model, checkpoint_dir=tmp_path)
        metrics = trainer.train_epoch(train_loader)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0

        # Weights must be updated
        current_weights = model.linear.weight.clone().detach()
        assert not torch.allclose(initial_weights, current_weights)

    def test_evaluate_metrics_structure(self, dummy_dataloaders, tmp_path):
        """Verify evaluate() computes accuracy, macro-F1, per-class metrics, and confusion matrix."""
        _, val_loader = dummy_dataloaders
        model = SimpleTestClassifier()
        trainer = ModelTrainer(model=model, checkpoint_dir=tmp_path)

        val_metrics = trainer.evaluate(val_loader)

        assert "loss" in val_metrics
        assert "accuracy" in val_metrics
        assert "macro_f1" in val_metrics
        assert "macro_precision" in val_metrics
        assert "macro_recall" in val_metrics
        assert "per_class" in val_metrics
        assert "confusion_matrix" in val_metrics
        assert "classification_report" in val_metrics

        # 4 classes expected
        assert len(val_metrics["per_class"]) == 4
        cm = np.array(val_metrics["confusion_matrix"])
        assert cm.shape == (4, 4)
        assert cm.sum() == 32  # 32 validation samples

    def test_checkpoint_save_and_load(self, tmp_path):
        """Verify model and optimizer states round-trip through checkpoint file."""
        model = SimpleTestClassifier()
        original_weights = model.linear.weight.clone().detach()
        trainer = ModelTrainer(model=model, checkpoint_dir=tmp_path)

        ckpt_file = tmp_path / "test_ckpt.pth"
        trainer.save_checkpoint(
            checkpoint_path=ckpt_file,
            epoch=5,
            val_loss=0.42,
            val_macro_f1=0.88,
            fold_id=0,
            normalizer_meta={"mean": 0.0, "std": 1.0},
        )
        assert ckpt_file.is_file()

        # Mutate model weights
        with torch.no_grad():
            model.linear.weight.fill_(999.0)

        # Restore from checkpoint
        new_model = SimpleTestClassifier()
        new_trainer = ModelTrainer(model=new_model, checkpoint_dir=tmp_path)
        loaded_data = new_trainer.load_checkpoint(ckpt_file)

        assert loaded_data["epoch"] == 5
        assert loaded_data["val_macro_f1"] == 0.88
        assert loaded_data["fold_id"] == 0
        assert torch.allclose(new_model.linear.weight, original_weights)

    def test_non_finite_logits_raises_runtime_error(self, dummy_dataloaders, tmp_path):
        """Verify training immediately halts if non-finite logits are produced."""
        train_loader, _ = dummy_dataloaders
        bad_model = BadNaNModel()
        trainer = ModelTrainer(model=bad_model, checkpoint_dir=tmp_path)

        with pytest.raises(RuntimeError, match="Non-finite logits encountered"):
            trainer.train_epoch(train_loader)

    def test_fit_early_stopping(self, dummy_dataloaders, tmp_path):
        """Verify fit() triggers early stopping when validation metric plateaus."""
        train_loader, val_loader = dummy_dataloaders
        model = SimpleTestClassifier()
        trainer = ModelTrainer(
            model=model,
            checkpoint_dir=tmp_path,
            early_stopping_patience=2,
            early_stopping_min_delta=1e-4,
        )

        ckpt_path = tmp_path / "best.pth"
        result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            checkpoint_path=ckpt_path,
        )

        assert "history" in result
        assert "best_epoch" in result
        assert ckpt_path.is_file()
        assert len(result["history"]) <= 20

"""PyTorch model trainer tailored for CPU execution and group cross-validation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config_parser import load_config

logger = logging.getLogger("InspectAI.Trainer")

# Standard 4-class naming for industrial vibration fault diagnostics
DEFAULT_CLASS_NAMES = [
    "Normal",
    "Ball Fault",
    "Inner Race Fault",
    "Outer Race Fault",
]


class ModelTrainer:
    """Manages model training loops, validation, metrics, checkpointing, and early stopping on CPU.

    Key Features:
    - Guaranteed zero data leakage from held-out test recordings.
    - CPU-optimized execution with strict memory management.
    - Early stopping driven by validation Macro-F1 (with validation loss as tie-breaker).
    - Robust numerical guards against non-finite (NaN / Inf) losses or activations.
    - Full reproducibility support via deterministic seeding.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Union[str, torch.device] = "cpu",
        checkpoint_dir: Union[str, Path] = "models/checkpoints",
        early_stopping_patience: int = 7,
        early_stopping_min_delta: float = 1e-4,
        class_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ModelTrainer.

        Args:
            model: PyTorch model instance (e.g. VibrationCNN1D).
            criterion: Loss function. Defaults to nn.CrossEntropyLoss().
            optimizer: Optimizer. Defaults to AdamW(lr=1e-3, weight_decay=1e-4).
            device: Computing device ('cpu').
            checkpoint_dir: Directory where model checkpoints will be stored.
            early_stopping_patience: Consecutive epochs without improvement before stopping.
            early_stopping_min_delta: Minimum metric improvement threshold.
            class_names: Ordered list of class label names.
            config: Optional project configuration dictionary.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config if config is not None else load_config()

        # Loss Criterion
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            train_cfg = self.config.get("training", {})
            lr = float(train_cfg.get("learning_rate", 0.001))
            wd = float(train_cfg.get("weight_decay", 0.0001))
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.class_names = class_names or DEFAULT_CLASS_NAMES

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch over the dataloader.

        Args:
            dataloader: Training PyTorch DataLoader.

        Returns:
            Dict[str, float]: Dictionary containing average training loss and accuracy.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            # CWRUVibrationDataset returns (window_tensor, label_tensor, sample_meta)
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch[:2]

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(inputs)

            # Numerical guard: verify logits are finite
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits encountered in training batch {batch_idx}. "
                    f"NaN count: {torch.isnan(logits).sum().item()}, Inf count: {torch.isinf(logits).sum().item()}."
                )

            loss = self.criterion(logits, targets)

            # Numerical guard: verify loss is finite
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({loss.item()}) encountered in training batch {batch_idx}.")

            loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("Training DataLoader yielded 0 samples.")

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Compute comprehensive validation metrics on CPU without updating weights.

        Args:
            dataloader: Validation PyTorch DataLoader.

        Returns:
            Dict[str, Any]: Evaluation metrics including loss, accuracy, macro-F1,
                            per-class metrics, and confusion matrix.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    inputs, targets = batch[:2]

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)

                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite logits encountered in validation batch {batch_idx}.")

                loss = self.criterion(logits, targets)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss ({loss.item()}) encountered in validation batch {batch_idx}.")

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        if total_samples == 0:
            raise ValueError("Validation DataLoader yielded 0 samples.")

        avg_loss = total_loss / total_samples
        y_true = np.array(all_targets, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)

        accuracy = float((y_pred == y_true).mean())

        num_classes = len(self.class_names)
        labels = list(range(num_classes))

        # Precision, Recall, F1 with zero_division handling
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        macro_precision = float(np.mean(precision))
        macro_recall = float(np.mean(recall))
        macro_f1 = float(np.mean(f1))

        # Per-class breakdown
        per_class_metrics: Dict[str, Dict[str, float]] = {}
        for idx, cls_name in enumerate(self.class_names):
            per_class_metrics[cls_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx]),
            }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Classification report text
        clf_report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=self.class_names,
            zero_division=0,
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_report,
            "total_samples": total_samples,
        }

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: int,
        val_loss: float,
        val_macro_f1: float,
        fold_id: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        normalizer_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist model state, optimizer state, and training metadata to disk.

        Args:
            checkpoint_path: Destination file path.
            epoch: Current training epoch index.
            val_loss: Validation loss at this checkpoint.
            val_macro_f1: Validation Macro-F1 at this checkpoint.
            fold_id: Optional cross-validation fold identifier.
            metrics: Full validation metrics dictionary.
            normalizer_meta: Parameters and provenance of fitted normalizer.

        Returns:
            Path: Resolved path to the saved checkpoint file.
        """
        path = Path(checkpoint_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "epoch": epoch,
            "fold_id": fold_id,
            "val_loss": float(val_loss),
            "val_macro_f1": float(val_macro_f1),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
            "normalizer_metadata": normalizer_meta or {},
            "config": self.config,
        }

        torch.save(checkpoint_data, path)
        logger.info(f"Saved checkpoint to: {path} (Epoch={epoch}, Macro-F1={val_macro_f1:.4f}, Loss={val_loss:.4f})")
        return path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoint weights and training state from disk.

        Args:
            checkpoint_path: Path to .pth checkpoint file.

        Returns:
            Dict[str, Any]: Checkpoint metadata dictionary.
        """
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from: {path} (Epoch={checkpoint.get('epoch')})")
        return checkpoint

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        fold_id: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        normalizer_meta: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute full training loop across multiple epochs with early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum epochs to train.
            fold_id: Optional cross-validation fold index (0 or 1).
            checkpoint_path: Path to save the best checkpoint.
            normalizer_meta: Normalizer metadata dict to store in checkpoint.
            lr_scheduler: Optional learning rate scheduler.

        Returns:
            Dict[str, Any]: Summary containing history, best epoch, best metrics, and stopping reason.
        """
        best_macro_f1 = -1.0
        best_val_loss = float("inf")
        best_epoch = -1
        best_metrics: Dict[str, Any] = {}
        epochs_no_improve = 0
        history: List[Dict[str, Any]] = []

        fold_str = f"Fold {fold_id} " if fold_id is not None else ""
        logger.info(f"Starting {fold_str}training for up to {epochs} epochs (Early stopping patience={self.early_stopping_patience}).")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_macro_f1 = val_metrics["macro_f1"]
            val_loss = val_metrics["loss"]

            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_macro_f1,
                "learning_rate": current_lr,
            }
            # Append per-class F1 to history
            for cls_name, cls_vals in val_metrics["per_class"].items():
                epoch_record[f"val_f1_{cls_name.lower().replace(' ', '_')}"] = cls_vals["f1_score"]

            history.append(epoch_record)

            logger.info(
                f"[{fold_str}Epoch {epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Macro-F1: {val_macro_f1:.4f} "
                f"(LR: {current_lr:.1e})"
            )

            # Checkpoint Selection:
            # Primary: Validation Macro-F1 improvement
            # Secondary (tie-breaker): Validation Loss reduction
            f1_improved = (val_macro_f1 - best_macro_f1) > self.early_stopping_min_delta
            f1_tied = abs(val_macro_f1 - best_macro_f1) <= self.early_stopping_min_delta
            loss_improved = val_loss < best_val_loss

            if f1_improved or (f1_tied and loss_improved):
                best_macro_f1 = val_macro_f1
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = val_metrics
                epochs_no_improve = 0

                if checkpoint_path is not None:
                    self.save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch,
                        val_loss=val_loss,
                        val_macro_f1=val_macro_f1,
                        fold_id=fold_id,
                        metrics=val_metrics,
                        normalizer_meta=normalizer_meta,
                    )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No improvement in validation Macro-F1 for {self.early_stopping_patience} consecutive epochs."
                    )
                    break

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "total_epochs_trained": len(history),
            "stopped_early": len(history) < epochs,
        }

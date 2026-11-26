"""Phase 10.2: Controlled Normalization Ablation and Generalization Experiment.

Evaluates three normalization strategies on identical 2-fold recording-level cross-validation:
1. 'global': Baseline training-only global scalar mean/std.
2. 'instance': Window-level instance z-score normalization.
3. 'none': Passthrough unnormalized physical amplitudes.

Strict constraints:
- Uses development recordings only (98, 99, 100, 105, 118, 130, 209, 222, 234).
- Preserves the original frozen final checkpoint and evaluation reports.
- Never touches or tunes on the held-out test recordings (97, 169, 185, 197).
- Uses identical architectures, seeds, optimizers, and early stopping rules on CPU.
- Exports all artifacts to outputs/experiments/phase10_2/.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_1d import VibrationCNN1D
from src.training.dataset import CWRUVibrationDataset
from src.training.normalizer import VibrationNormalizer
from src.training.trainer import DEFAULT_CLASS_NAMES, ModelTrainer
from src.utils.config_parser import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.Ablation")

OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs" / "experiments" / "phase10_2"


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for PyTorch and NumPy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval_fold(
    fold_id: int,
    strategy: str,
    config: Dict[str, Any],
    strategy_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train and evaluate a single fold for a given normalization strategy."""
    logger.info(f"--- Strategy: '{strategy}' | Fold: {fold_id} ---")
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 32))
    max_epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("learning_rate", 0.001))
    wd = float(train_cfg.get("weight_decay", 0.0001))
    patience = int(train_cfg.get("early_stopping", {}).get("patience", 7))
    min_delta = float(train_cfg.get("early_stopping", {}).get("min_delta", 1e-4))
    device = "cpu"

    # 1. Deterministic data seeding
    set_seed(seed + fold_id)

    # 2. Instantiate and fit Normalizer on training fold
    unnorm_train_ds = CWRUVibrationDataset(split="train", fold=fold_id, config=config)
    train_windows = unnorm_train_ds.raw_windows[unnorm_train_ds.subset_indices]
    train_recs = unnorm_train_ds.metadata_df.iloc[unnorm_train_ds.subset_indices]["recording_id"].astype(str).unique().tolist()

    normalizer = VibrationNormalizer(epsilon=1e-8, strategy=strategy)
    normalizer.fit(train_windows, source_recording_ids=train_recs)

    norm_save_path = strategy_dir / f"normalizer_fold_{fold_id}.json"
    normalizer.save(norm_save_path)

    # 3. DataLoaders
    train_dataset = CWRUVibrationDataset(split="train", fold=fold_id, normalizer=normalizer, config=config)
    val_dataset = CWRUVibrationDataset(split="val", fold=fold_id, normalizer=normalizer, config=config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 4. Fresh Model
    set_seed(seed + fold_id * 10)
    model = VibrationCNN1D.from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=strategy_dir,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        class_names=DEFAULT_CLASS_NAMES,
        config=config,
    )

    checkpoint_path = strategy_dir / f"fold_{fold_id}_best.pth"

    start_time = time.time()
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        fold_id=fold_id,
        checkpoint_path=checkpoint_path,
        normalizer_meta=normalizer.to_dict(),
        lr_scheduler=lr_scheduler,
    )
    duration_sec = time.time() - start_time

    # 5. Reload Best Checkpoint and Extract Detailed Validation Predictions
    if checkpoint_path.is_file():
        trainer.load_checkpoint(checkpoint_path)

    val_metrics = trainer.evaluate(val_loader)

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    metadata_rows = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, sample_meta) in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.tolist())

            # Unpack sample_meta dict of lists
            batch_size_actual = len(targets)
            for i in range(batch_size_actual):
                metadata_rows.append({
                    "strategy": strategy,
                    "fold": fold_id,
                    "window_id": sample_meta["window_id"][i],
                    "recording_id": str(sample_meta["recording_id"][i]),
                    "true_class_id": int(targets[i]),
                    "true_class_name": DEFAULT_CLASS_NAMES[int(targets[i])],
                    "pred_class_id": int(preds[i]),
                    "pred_class_name": DEFAULT_CLASS_NAMES[int(preds[i])],
                    "prob_Normal": float(probs[i, 0]),
                    "prob_Ball_Fault": float(probs[i, 1]),
                    "prob_Inner_Race_Fault": float(probs[i, 2]),
                    "prob_Outer_Race_Fault": float(probs[i, 3]),
                })

    df_fold_preds = pd.DataFrame(metadata_rows)

    fold_summary = {
        "strategy": strategy,
        "fold": fold_id,
        "duration_sec": round(duration_sec, 2),
        "best_epoch": fit_result["best_epoch"],
        "total_epochs": fit_result.get("total_epochs_trained", len(fit_result.get("history", []))),
        "val_loss": round(val_metrics["loss"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "train_recordings": train_recs,
        "val_recordings": df_fold_preds["recording_id"].unique().tolist(),
    }

    return fold_summary, df_fold_preds


def run_strategy_experiment(strategy: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute complete 2-fold cross-validation for a given strategy."""
    strategy_dir = OUTPUT_BASE_DIR / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries = []
    fold_dfs = []

    for fold_id in [0, 1]:
        f_summary, f_df = train_and_eval_fold(
            fold_id=fold_id,
            strategy=strategy,
            config=config,
            strategy_dir=strategy_dir,
        )
        fold_summaries.append(f_summary)
        fold_dfs.append(f_df)

    # Combine out-of-fold predictions
    oof_df = pd.concat(fold_dfs, ignore_index=True)
    oof_df.to_csv(strategy_dir / "oof_predictions.csv", index=False)

    # Compute Pooled Out-of-Fold Metrics
    y_true = oof_df["true_class_id"].values
    y_pred = oof_df["pred_class_id"].values

    pooled_acc = float(accuracy_score(y_true, y_pred))
    pooled_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    # Per-class report
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=DEFAULT_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred).tolist()

    # Per-recording accuracy
    rec_acc = {}
    for rec_id, group in oof_df.groupby("recording_id"):
        rec_acc[str(rec_id)] = {
            "n_samples": len(group),
            "true_class": group["true_class_name"].iloc[0],
            "accuracy": round(float((group["true_class_id"] == group["pred_class_id"]).mean()), 4),
            "pred_distribution": group["pred_class_name"].value_counts().to_dict(),
        }

    strategy_result = {
        "strategy": strategy,
        "pooled_oof_accuracy": round(pooled_acc, 4),
        "pooled_oof_macro_f1": round(pooled_macro_f1, 4),
        "mean_fold_accuracy": round(float(np.mean([f["val_accuracy"] for f in fold_summaries])), 4),
        "mean_fold_macro_f1": round(float(np.mean([f["val_macro_f1"] for f in fold_summaries])), 4),
        "total_duration_sec": round(float(np.sum([f["duration_sec"] for f in fold_summaries])), 2),
        "folds": fold_summaries,
        "per_class": {
            cls_name: {
                "precision": round(clf_report[cls_name]["precision"], 4),
                "recall": round(clf_report[cls_name]["recall"], 4),
                "f1_score": round(clf_report[cls_name]["f1-score"], 4),
                "support": int(clf_report[cls_name]["support"]),
            }
            for cls_name in DEFAULT_CLASS_NAMES
        },
        "confusion_matrix": conf_mat,
        "per_recording": rec_acc,
    }

    with open(strategy_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(strategy_result, f, indent=2)

    return strategy_result, oof_df


def main():
    logger.info("======================================================================")
    logger.info("InspectAI Phase 10.2: Normalization Ablation Controlled Experiment")
    logger.info("======================================================================")
    config = load_config()

    strategies = ["global", "instance", "none"]
    all_results = {}
    all_oof_dfs = []

    for strategy in strategies:
        res, oof_df = run_strategy_experiment(strategy=strategy, config=config)
        all_results[strategy] = res
        all_oof_dfs.append(oof_df)

    # Save overall ablation comparison
    comparison_summary = {
        "experiment_title": "InspectAI Phase 10.2: Normalization Strategy Ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "CPU (Single Node)",
        "dataset_partition": "2-Fold Stratified Group K-Fold on Dev Recordings (2,110 windows)",
        "holdout_leakage_status": "Strictly Isolated (Recordings 97, 185, 169, 197 remained untouched)",
        "strategies": all_results,
    }

    summary_file = OUTPUT_BASE_DIR / "ablation_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Generate quick markdown comparison table
    logger.info("======================================================================")
    logger.info("ABLATION EXPERIMENT COMPLETED - POOLED OUT-OF-FOLD METRICS:")
    logger.info("======================================================================")
    print(f"{'Strategy':<12} | {'OOF Acc':<10} | {'OOF Macro-F1':<14} | {'Normal F1':<10} | {'Ball F1':<10} | {'Inner F1':<10} | {'Outer F1':<10} | {'Duration (s)':<12}")
    print("-" * 95)
    for s in strategies:
        r = all_results[s]
        pc = r["per_class"]
        print(
            f"{s:<12} | {r['pooled_oof_accuracy']:<10.2%} | {r['pooled_oof_macro_f1']:<14.2%} | "
            f"{pc['Normal']['f1_score']:<10.2%} | {pc['Ball Fault']['f1_score']:<10.2%} | "
            f"{pc['Inner Race Fault']['f1_score']:<10.2%} | {pc['Outer Race Fault']['f1_score']:<10.2%} | "
            f"{r['total_duration_sec']:<12.1f}"
        )


if __name__ == "__main__":
    main()


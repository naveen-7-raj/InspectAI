"""Phase 10.3: Controlled Architecture Experiment - Baseline 1D CNN vs. Multiscale 1D CNN.

Evaluates:
- Baseline: VibrationCNN1D (73,156 parameters) with global normalization.
- Multiscale: MultiscaleVibrationCNN1D (81,252 parameters, kernels [5, 15, 65]) with global normalization.

Controlled experimental guarantees:
- Evaluates strictly on the 9 development recordings (2,110 windows) via 2-fold group CV.
- Preserves the original final model checkpoint and all previous evaluation reports.
- Held-out test recordings (97, 169, 185, 197) remain untouched.
- Identical seeds, optimizers, early-stopping rules, and CPU hardware.
- Exports all artifacts to outputs/experiments/phase10_3/.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_1d import VibrationCNN1D
from src.models.multiscale_cnn_1d import MultiscaleVibrationCNN1D
from src.training.dataset import CWRUVibrationDataset
from src.training.normalizer import VibrationNormalizer
from src.training.trainer import DEFAULT_CLASS_NAMES, ModelTrainer
from src.utils.config_parser import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.MultiscaleAblation")

OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs" / "experiments" / "phase10_3"


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure_cpu_latency(model: torch.nn.Module, n_runs: int = 100) -> float:
    """Measure single-window CPU inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 1, 2048)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.median(latencies))


def train_and_eval_architecture_fold(
    fold_id: int,
    arch_name: str,
    model_factory,
    config: Dict[str, Any],
    arch_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train and evaluate a single fold for a chosen architecture."""
    logger.info(f"--- Architecture: '{arch_name}' | Fold: {fold_id} ---")
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 32))
    max_epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("learning_rate", 0.001))
    wd = float(train_cfg.get("weight_decay", 0.0001))
    patience = int(train_cfg.get("early_stopping", {}).get("patience", 7))
    min_delta = float(train_cfg.get("early_stopping", {}).get("min_delta", 1e-4))
    device = "cpu"

    set_seed(seed + fold_id)

    # 1. Fit Normalizer on training fold strictly
    unnorm_train_ds = CWRUVibrationDataset(split="train", fold=fold_id, config=config)
    train_windows = unnorm_train_ds.raw_windows[unnorm_train_ds.subset_indices]
    train_recs = unnorm_train_ds.metadata_df.iloc[unnorm_train_ds.subset_indices]["recording_id"].astype(str).unique().tolist()

    normalizer = VibrationNormalizer(epsilon=1e-8, strategy="global")
    normalizer.fit(train_windows, source_recording_ids=train_recs)

    norm_save_path = arch_dir / f"normalizer_fold_{fold_id}.json"
    normalizer.save(norm_save_path)

    # 2. DataLoaders
    train_dataset = CWRUVibrationDataset(split="train", fold=fold_id, normalizer=normalizer, config=config)
    val_dataset = CWRUVibrationDataset(split="val", fold=fold_id, normalizer=normalizer, config=config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 3. Model Instantiation
    set_seed(seed + fold_id * 10)
    model = model_factory()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=arch_dir,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        class_names=DEFAULT_CLASS_NAMES,
        config=config,
    )

    checkpoint_path = arch_dir / f"fold_{fold_id}_best.pth"

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

    # 4. Reload Best Checkpoint and Evaluate
    if checkpoint_path.is_file():
        trainer.load_checkpoint(checkpoint_path)

    val_metrics = trainer.evaluate(val_loader)
    latency_ms = measure_cpu_latency(model)

    # Extract Out-of-Fold Predictions
    model.eval()
    metadata_rows = []
    with torch.no_grad():
        for inputs, targets, sample_meta in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            for i in range(len(targets)):
                metadata_rows.append({
                    "architecture": arch_name,
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
        "architecture": arch_name,
        "fold": fold_id,
        "parameters": param_count,
        "duration_sec": round(duration_sec, 2),
        "best_epoch": fit_result["best_epoch"],
        "total_epochs": fit_result.get("total_epochs_trained", len(fit_result.get("history", []))),
        "val_loss": round(val_metrics["loss"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "cpu_latency_ms": round(latency_ms, 3),
        "train_recordings": train_recs,
        "val_recordings": df_fold_preds["recording_id"].unique().tolist(),
    }

    return fold_summary, df_fold_preds


def run_architecture_experiment(
    arch_name: str,
    model_factory,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute complete 2-fold group cross-validation for a given model architecture."""
    arch_dir = OUTPUT_BASE_DIR / arch_name
    arch_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries = []
    fold_dfs = []

    for fold_id in [0, 1]:
        f_summary, f_df = train_and_eval_architecture_fold(
            fold_id=fold_id,
            arch_name=arch_name,
            model_factory=model_factory,
            config=config,
            arch_dir=arch_dir,
        )
        fold_summaries.append(f_summary)
        fold_dfs.append(f_df)

    oof_df = pd.concat(fold_dfs, ignore_index=True)
    oof_df.to_csv(arch_dir / "oof_predictions.csv", index=False)

    y_true = oof_df["true_class_id"].values
    y_pred = oof_df["pred_class_id"].values

    pooled_acc = float(accuracy_score(y_true, y_pred))
    pooled_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=DEFAULT_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred).tolist()

    rec_breakdown = {}
    for rec_id, group in oof_df.groupby("recording_id"):
        rec_breakdown[str(rec_id)] = {
            "n_samples": len(group),
            "true_class": group["true_class_name"].iloc[0],
            "accuracy": round(float((group["true_class_id"] == group["pred_class_id"]).mean()), 4),
            "pred_distribution": group["pred_class_name"].value_counts().to_dict(),
        }

    arch_result = {
        "architecture": arch_name,
        "parameter_count": fold_summaries[0]["parameters"],
        "pooled_oof_accuracy": round(pooled_acc, 4),
        "pooled_oof_macro_f1": round(pooled_macro_f1, 4),
        "mean_fold_accuracy": round(float(np.mean([f["val_accuracy"] for f in fold_summaries])), 4),
        "mean_fold_macro_f1": round(float(np.mean([f["val_macro_f1"] for f in fold_summaries])), 4),
        "mean_cpu_latency_ms": round(float(np.mean([f["cpu_latency_ms"] for f in fold_summaries])), 3),
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
        "per_recording": rec_breakdown,
    }

    with open(arch_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(arch_result, f, indent=2)

    return arch_result, oof_df


def main():
    logger.info("======================================================================")
    logger.info("InspectAI Phase 10.3: Multiscale 1D CNN Architecture Controlled Experiment")
    logger.info("======================================================================")
    config = load_config()

    architectures = [
        ("baseline_1d_cnn", lambda: VibrationCNN1D.from_config(config)),
        ("multiscale_1d_cnn", lambda: MultiscaleVibrationCNN1D.from_config(config)),
    ]

    results = {}
    for arch_name, factory in architectures:
        res, _ = run_architecture_experiment(arch_name=arch_name, model_factory=factory, config=config)
        results[arch_name] = res

    summary_file = OUTPUT_BASE_DIR / "multiscale_comparison_summary.json"
    comparison = {
        "experiment_title": "InspectAI Phase 10.3: Baseline vs. Multiscale 1D CNN Architecture",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "CPU (Single Node)",
        "dataset_partition": "2-Fold Stratified Group K-Fold on Dev Recordings (2,110 windows)",
        "holdout_leakage_status": "Strictly Isolated (Recordings 97, 185, 169, 197 remained untouched)",
        "architectures": results,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logger.info("======================================================================")
    logger.info("ARCHITECTURE ABLATION COMPLETED - POOLED OUT-OF-FOLD METRICS:")
    logger.info("======================================================================")
    print(f"{'Architecture':<20} | {'Params':<8} | {'OOF Acc':<10} | {'OOF Macro-F1':<14} | {'Normal F1':<10} | {'Ball F1':<10} | {'Inner F1':<10} | {'Outer F1':<10} | {'Latency':<9} | {'Train(s)':<8}")
    print("-" * 125)
    for arch_name, _ in architectures:
        r = results[arch_name]
        pc = r["per_class"]
        print(
            f"{arch_name:<20} | {r['parameter_count']:<8} | {r['pooled_oof_accuracy']:<10.2%} | {r['pooled_oof_macro_f1']:<14.2%} | "
            f"{pc['Normal']['f1_score']:<10.2%} | {pc['Ball Fault']['f1_score']:<10.2%} | "
            f"{pc['Inner Race Fault']['f1_score']:<10.2%} | {pc['Outer Race Fault']['f1_score']:<10.2%} | "
            f"{r['mean_cpu_latency_ms']:<6.2f} ms | {r['total_duration_sec']:<8.1f}"
        )


if __name__ == "__main__":
    main()


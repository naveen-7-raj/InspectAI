import json
from pathlib import Path
import torch

exp_dir = Path("outputs/experiments/phase11_4")
ckpts_dir = exp_dir / "checkpoints"

with open(exp_dir / "cross_fold_comparison.json") as f:
    cross = json.load(f)

with open(exp_dir / "fold_metrics.json") as f:
    folds = json.load(f)

with open(exp_dir / "model_parameters.json") as f:
    params = json.load(f)

print("=== EXPERIMENTS IN CROSS FOLD COMPARISON ===")
for c in cross:
    eid = c["experiment_id"]
    print(f"{eid}: pooled_macro_f1={c['pooled_macro_f1']}, pooled_acc={c['pooled_accuracy']}, rec172_acc={c['recording_172_accuracy']}, params={c['parameters']}")

print("\n=== CHECKPOINT VALIDATION ===")
for c in cross:
    eid = c["experiment_id"]
    for f_idx in [1, 2, 3]:
        ckpt_file = ckpts_dir / f"{eid}_fold_{f_idx}.pth"
        assert ckpt_file.exists(), f"Missing {ckpt_file}"
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        print(f"{ckpt_file.name}: best_epoch={ckpt.get('best_epoch')}, is_dual={ckpt.get('is_dual')}, keys={list(ckpt.keys())}")

print("\n=== EXPERIMENT B FOLD METRICS DETAIL ===")
exp_b_folds = folds["experiment_b"]["folds"]
for f in exp_b_folds:
    print(f"Fold {f['fold']} (held-out {f['held_out_load_hp']} HP): acc={f['accuracy']}, f1={f['macro_f1']}, best_epoch={f['best_epoch']}, latency={f['latency_ms']['single_window_median_ms']} ms")
    if f["fold"] == 3:
        print(f"  Rec 172 in Fold 3: {f['recording_172']}")


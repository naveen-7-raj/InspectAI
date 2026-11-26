"""Technical consistency audit script for Phase 22."""
import json
import hashlib
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.models.spectral_fusion_cnn import SpectralFusionCNN
from src.inference.predictor import StreamingPredictor

def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def audit():
    print("==================================================")
    print("INSPECTAI TECHNICAL CONSISTENCY AUDIT (PHASE 22)")
    print("==================================================")

    # 1. Model & Checksums
    ckpt_path = Path("models/final/inspectai_final_model.pth")
    ckpt_hash = get_sha256(ckpt_path)
    print(f"\n1. MODEL CHECKPOINT:")
    print(f"   Path: {ckpt_path}")
    print(f"   SHA-256: {ckpt_hash}")
    print(f"   Size: {ckpt_path.stat().st_size} bytes ({ckpt_path.stat().st_size / 1024:.2f} KB)")

    predictor = StreamingPredictor.from_final_model("models/final")
    model = predictor.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model Class: {type(model).__name__}")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")

    # Submodule breakdown
    wave_params = sum(p.numel() for p in model.wave_block1.parameters()) + \
                  sum(p.numel() for p in model.wave_block2.parameters()) + \
                  sum(p.numel() for p in model.conv3.parameters()) + \
                  sum(p.numel() for p in model.bn3.parameters())
    spec_params = sum(p.numel() for p in model.spectral_branch.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"   - Temporal Branch Parameters: {wave_params:,} ({wave_params/total_params*100:.1f}%)")
    print(f"   - Spectral Branch Parameters: {spec_params:,} ({spec_params/total_params*100:.1f}%)")
    print(f"   - Dense Classifier Head:      {cls_params:,} ({cls_params/total_params*100:.1f}%)")

    # 2. Normalizers
    print(f"\n2. NORMALIZERS:")
    norm_w = Path("models/final/inspectai_final_normalizer.json")
    norm_s = Path("models/final/inspectai_final_spectral_normalizer.json")
    print(f"   Waveform Normalizer: {norm_w.name} | SHA-256: {get_sha256(norm_w)}")
    print(f"   Spectral Normalizer: {norm_s.name} | SHA-256: {get_sha256(norm_s)}")

    # 3. Benchmark Metrics
    print(f"\n3. CWRU TEST SET PERFORMANCE (Phase 11.5 / 12):")
    final_sum_path = Path("outputs/experiments/phase11_5/final_model_summary.json")
    if final_sum_path.exists():
        with open(final_sum_path) as f:
            cwru_data = json.load(f)
        eval_metrics = cwru_data.get("evaluation_metrics", {})
        print(f"   Test Accuracy:        {eval_metrics.get('test_accuracy', 0)*100:.2f}%")
        print(f"   Test Macro-F1:        {eval_metrics.get('macro_f1', 0)*100:.2f}%")
        print(f"   Test Balanced Acc:    {eval_metrics.get('balanced_accuracy', 0)*100:.2f}%")
        print(f"   Protected Holdouts:   {cwru_data.get('protected_holdout_recordings', [])}")

    print(f"\n4. PADERBORN ZERO-SHOT VS CORAL PERFORMANCE:")
    p17_gen_path = Path("outputs/experiments/phase17/generalization_summary.json")
    if p17_gen_path.exists():
        with open(p17_gen_path) as f:
            p17_data = json.load(f)
        print("   Per Condition Comparison (Disjoint 160 Unseen Recordings, 14,436 windows):")
        for cond in p17_data.get("per_condition_comparison", []):
            print(f"   - Condition {cond['condition']} ({cond['speed_rpm']} RPM):")
            print(f"     Baseline Window BalAcc: {cond['baseline_win_bal_acc']:.2f}% | Recording BalAcc: {cond['baseline_rec_bal_acc']:.2f}%")
            print(f"     Disjoint CORAL Win BalAcc: {cond['disjoint_coral_win_bal_acc']:.2f}% | Recording BalAcc: {cond['disjoint_coral_rec_bal_acc']:.2f}% | Win F1: {cond['disjoint_coral_win_f1']:.2f}%")
            print(f"     Healthy (Normal) Recall: {cond['normal_rec_recall']:.1f}% | Inner Race Recall: {cond['inner_rec_recall']:.1f}%")
        
        agg = p17_data.get("aggregate_metrics", {})
        print("\n   Aggregate Across All 4 Operating Conditions:")
        print(f"     Mean Baseline Window BalAcc:     {agg.get('mean_baseline_window_bal_acc', 0):.2f}%")
        print(f"     Mean Baseline Recording BalAcc:  {agg.get('mean_baseline_recording_bal_acc', 0):.2f}%")
        print(f"     Mean Disjoint CORAL Window BalAcc: {agg.get('mean_disjoint_coral_window_bal_acc', 0):.2f}%")
        print(f"     Mean Disjoint CORAL Rec BalAcc:    {agg.get('mean_disjoint_coral_recording_bal_acc', 0):.2f}%")
        print(f"     Mean Transductive Gap:             {agg.get('mean_transductive_gap_pct', 0):.2f}%")

    print(f"\n5. LATENCY, THROUGHPUT & HEADROOM:")
    # Benchmark latency on CPU
    x_test = np.random.randn(2048).astype(np.float32)
    # Warmup
    for _ in range(20):
        predictor.predict_window(x_test, mode="baseline")
    
    import time
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        predictor.predict_window(x_test, mode="baseline")
        latencies.append((time.perf_counter() - t0) * 1000)
    
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    print(f"   Single-window inference latency: Mean = {mean_lat:.2f} ms | P95 = {p95_lat:.2f} ms")
    hop_size = 512
    fs = 12000
    hop_duration_ms = (hop_size / fs) * 1000
    headroom = (hop_duration_ms - mean_lat) / hop_duration_ms * 100
    print(f"   Streaming Window Hop Duration:   {hop_duration_ms:.2f} ms ({hop_size} samples @ {fs} Hz)")
    print(f"   Real-time Headroom:              {headroom:.1f}% (Inference uses only {mean_lat/hop_duration_ms*100:.1f}% of budget)")

    print("\n==================================================")

if __name__ == "__main__":
    audit()

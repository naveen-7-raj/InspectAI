"""Generate dark-themed model results asset for portfolio video editing."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "video_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_video_results_asset():
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#0F172A")  # Slate 900

    # Grid specification
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25,
                           top=0.88, bottom=0.10, left=0.08, right=0.92)

    # Main Title
    fig.text(0.5, 0.95, "InspectAI: Empirical Evaluation & Domain Sensitivity Analysis",
             fontsize=20, weight="bold", color="#F8FAFC", ha="center")
    fig.text(0.5, 0.91, "CWRU Bearing Benchmark: 9 Development Recordings vs. 4 Untouched Holdout Recordings",
             fontsize=12, color="#94A3B8", ha="center")

    # Panel 1: Split-by-Split Performance Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1E293B")
    splits = ["Training Split\n(9 Dev Recs)", "2-Fold Group CV\n(Mean OOF)", "Holdout Test\n(4 Untouched Recs)"]
    macro_f1s = [99.95, 84.77, 34.04]
    accuracies = [99.95, 88.58, 39.73]
    x = np.arange(len(splits))
    width = 0.32

    r1 = ax1.bar(x - width/2, accuracies, width, label="Accuracy (%)", color="#38BDF8", edgecolor="#0284C7", linewidth=1.2)
    r2 = ax1.bar(x + width/2, macro_f1s, width, label="Macro-F1 (%)", color="#F59E0B", edgecolor="#D97706", linewidth=1.2)

    ax1.set_title("Performance Collapse Under Recording-Level Isolation", fontsize=13, weight="bold", color="#F1F5F9", pad=10)
    ax1.set_ylabel("Score (%)", fontsize=11, color="#CBD5E1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=10, color="#E2E8F0")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#64748B")
    ax1.legend(loc="upper right", framealpha=0.8, facecolor="#0F172A", edgecolor="#334155")

    for rects in [r1, r2]:
        for r in rects:
            h = r.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                         fontsize=10, weight="bold", color="#F8FAFC")

    # Panel 2: Class-by-Class Holdout F1-Scores (Horizontal Bar Chart)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1E293B")
    classes = ["Normal Baseline", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
    f1_scores = [90.27, 45.88, 0.00, 0.00]
    colors = ["#10B981", "#F59E0B", "#EF4444", "#EF4444"]

    y_pos = np.arange(len(classes))
    h_bars = ax2.barh(y_pos, f1_scores, color=colors, edgecolor="#334155", height=0.55, linewidth=1.2)
    ax2.set_title("Holdout Test F1-Score per Defect Class (818 Windows)", fontsize=13, weight="bold", color="#F1F5F9", pad=10)
    ax2.set_xlabel("F1-Score (%)", fontsize=11, color="#CBD5E1")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes, fontsize=10, color="#E2E8F0")
    ax2.set_xlim(0, 115)
    ax2.grid(axis="x", linestyle="--", alpha=0.25, color="#64748B")
    ax2.invert_yaxis()

    for bar, score in zip(h_bars, f1_scores):
        ax2.annotate(f"{score:.2f}%" if score > 0 else "0.00% (Collapse)",
                     xy=(score + 2, bar.get_y() + bar.get_height() / 2),
                     va="center", ha="left", fontsize=10, weight="bold",
                     color="#F8FAFC" if score > 0 else "#F87171")

    # Panel 3: Holdout Confusion Matrix Breakdown (Heatmap)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1E293B")
    cm = np.array([
        [116,   0,   0,   0],  # Normal (116 total)
        [ 25, 209,   0,   0],  # Ball Fault (234 total: 209 predicted Ball, 25 predicted Normal)
        [  0, 234,   0,   0],  # Inner Race Fault (234 total: 234 predicted Ball)
        [  0, 234,   0,   0]   # Outer Race Fault (234 total: 234 predicted Ball)
    ])
    im = ax3.imshow(cm, cmap="Blues", interpolation="nearest")
    ax3.set_title("Holdout Confusion Matrix (325 / 818 Correct)", fontsize=13, weight="bold", color="#F1F5F9", pad=10)
    tick_marks = np.arange(len(classes))
    short_labels = ["Normal", "Ball", "Inner", "Outer"]
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(short_labels, fontsize=10, color="#CBD5E1")
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(short_labels, fontsize=10, color="#CBD5E1")
    ax3.set_xlabel("Predicted Class", fontsize=11, color="#CBD5E1")
    ax3.set_ylabel("True Class", fontsize=11, color="#CBD5E1")

    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            txt_color = "#FFFFFF" if val > 100 else ("#94A3B8" if val == 0 else "#F8FAFC")
            ax3.text(j, i, f"{val}", ha="center", va="center", color=txt_color, fontsize=11, weight="bold")

    # Panel 4: Root Cause Engineering Diagnosis (Text Box)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    ax4.axis("off")

    summary_text = (
        "DIAGNOSTIC ROOT-CAUSE FINDINGS:\n\n"
        "1. Defect Geometry Domain Shift (Primary Hypothesis):\n"
        "   - Training split contained small (0.007\") and large (0.021\") race faults.\n"
        "   - Holdout test contained intermediate (0.014\") faults (169 & 197).\n"
        "   - Transient impulse shock energy fell between trained decision clusters.\n\n"
        "2. Spatial Load-Zone Variance (Secondary Hypothesis):\n"
        "   - Recording 197 outer race defect located at 6:00 (primary gravity load zone).\n"
        "   - Training recordings 130 & 234 located at 3:00 / 12:00 (orthogonal zones).\n\n"
        "3. Classification Head Default:\n"
        "   - Normal: 116/116 correct (100% recall). Ball: 209/234 correct (89.32% recall).\n"
        "   - 100% of unseen race windows (234 Inner, 234 Outer) defaulted to Ball Fault.\n\n"
        "INDUSTRIAL DISCLAIMER:\n"
        "Model requires domain adaptation and uncertainty gating prior to industrial use."
    )
    ax4.text(0.05, 0.95, summary_text, fontsize=9.5, color="#E2E8F0",
             family="monospace", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#0F172A", edgecolor="#EF4444", alpha=0.9))

    out_path = OUT_DIR / "model_evaluation_summary.png"
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated video results graphic: {out_path.name}")

if __name__ == "__main__":
    generate_video_results_asset()


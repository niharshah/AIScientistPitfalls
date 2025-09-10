import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

try:
    node = exp["no_projection_head"]["spr_bench"]
    tloss = node["losses"]["train"]
    vloss = node["losses"]["val"]
    metrics_val = node["metrics"]["val"]
    preds = node["predictions"]
    gts = node["ground_truth"]
except KeyError as e:
    print(f"Missing key in experiment data: {e}")
    exit()

epochs = range(1, len(tloss) + 1)

# 1) Train / Val loss curve
try:
    plt.figure()
    plt.plot(epochs, tloss, label="Train Loss")
    plt.plot(epochs, vloss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Validation metrics curve
try:
    swa = [m["swa"] for m in metrics_val]
    cwa = [m["cwa"] for m in metrics_val]
    ccwa = [m["ccwa"] for m in metrics_val]
    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, ccwa, label="CCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation Alignment Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_alignment_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# 3) Predicted vs Ground-Truth label distribution
try:
    if preds and gts:
        import collections

        gt_counter = collections.Counter(gts)
        pr_counter = collections.Counter(preds)
        labels = sorted(set(gt_counter.keys()) | set(pr_counter.keys()))
        gt_vals = [gt_counter[l] for l in labels]
        pr_vals = [pr_counter[l] for l in labels]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_vals, width, label="Ground Truth")
        plt.bar(x + width / 2, pr_vals, width, label="Predictions")
        plt.xlabel("Label ID")
        plt.ylabel("Count")
        plt.title("SPR_BENCH: Label Distribution (Val Best Epoch)")
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_label_distribution.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating label distribution plot: {e}")
    plt.close()

# Print best CCWA
try:
    best_ccwa = max(ccwa) if "ccwa" in locals() else None
    if best_ccwa is not None:
        print(f"Best validation CCWA: {best_ccwa:.4f}")
except Exception as e:
    print(f"Error computing best CCWA: {e}")

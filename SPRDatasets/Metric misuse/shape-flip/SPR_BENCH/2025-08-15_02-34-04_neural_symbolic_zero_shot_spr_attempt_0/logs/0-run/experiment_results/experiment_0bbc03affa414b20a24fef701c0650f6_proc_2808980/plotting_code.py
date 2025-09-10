import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("spr_bench", {})


# Helper to extract series safely
def get_series(key1, key2):
    return spr_data.get(key1, {}).get(key2, [])


# 1) Loss curve ---------------------------------------------------------------
try:
    train_loss = get_series("losses", "train")
    val_loss = get_series("losses", "val")
    if train_loss and val_loss:
        plt.figure()
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Train vs Val Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Metric curves ------------------------------------------------------------
try:
    tr_metrics = get_series("metrics", "train")  # list of (SWA,CWA,HWA)
    va_metrics = get_series("metrics", "val")
    if tr_metrics and va_metrics:
        tr_metrics = np.array(tr_metrics)  # shape [E,3]
        va_metrics = np.array(va_metrics)
        epochs = range(1, len(tr_metrics) + 1)
        plt.figure(figsize=(6, 4))
        labels = ["SWA", "CWA", "HWA"]
        for i, lbl in enumerate(labels):
            plt.plot(epochs, tr_metrics[:, i], label=f"Train {lbl}")
            plt.plot(epochs, va_metrics[:, i], linestyle="--", label=f"Val {lbl}")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH: Train/Val Metric Curves")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_metric_curves.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# 3) Final test metrics -------------------------------------------------------
try:
    test_metric = spr_data.get("metrics", {}).get("test", None)  # tuple (SWA,CWA,HWA)
    if test_metric:
        plt.figure()
        labels = ["SWA", "CWA", "HWA"]
        plt.bar(labels, test_metric, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        for i, v in enumerate(test_metric):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.title("SPR_BENCH: Final Test Metrics")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar plot: {e}")
    plt.close()

# Print test metrics to console
if test_metric:
    print(
        f"Test metrics -> SWA: {test_metric[0]:.4f}, CWA: {test_metric[1]:.4f}, HWA: {test_metric[2]:.4f}"
    )

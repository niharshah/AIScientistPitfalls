import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}


# ---------- helper ----------
def safe_get(path, default=None):
    ref = data
    for key in path:
        ref = ref.get(key, {})
    return ref if ref else default


loss_train = safe_get(["losses", "train"], [])
loss_val = safe_get(["losses", "val"], [])
metric_train = safe_get(["metrics", "train"], [])
metric_val = safe_get(["metrics", "val"], [])

epochs_loss = list(range(1, len(loss_train) + 1))
epochs_metric = list(range(1, len(metric_train) + 1))

# ---------- loss plot ----------
try:
    if loss_train and loss_val:
        plt.figure()
        plt.plot(epochs_loss, loss_train, label="Train Loss")
        plt.plot(epochs_loss, loss_val, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- metric plot ----------
try:
    if metric_train and metric_val:
        plt.figure()
        plt.plot(epochs_metric, metric_train, label="Train CWA2")
        plt.plot(epochs_metric, metric_val, label="Validation CWA2")
        plt.xlabel("Epoch")
        plt.ylabel("CWA2 Score")
        plt.title("SPR_BENCH CWA2 Metric Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa2_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

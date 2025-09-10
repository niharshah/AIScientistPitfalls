import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = {}


# helper for safe fetch
def g(path, default=None):
    cur = spr_data
    for k in path:
        if cur is None:
            return default
        cur = cur.get(k, None)
    return cur if cur is not None else default


# -------------------- Plot 1: Accuracy --------------------
try:
    epochs = range(1, len(g(["metrics", "train_acc"], [])) + 1)
    train_acc = g(["metrics", "train_acc"], [])
    val_acc = g(["metrics", "val_acc"], [])
    if train_acc and val_acc:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy per Epoch\nTrain vs. Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------------------- Plot 2: Loss --------------------
try:
    train_loss = g(["losses", "train"], [])
    val_loss = g(["losses", "val"], [])
    if train_loss and val_loss:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss per Epoch\nTrain vs. Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- Plot 3: Final Test Metrics --------------------
try:
    overall_acc = g(["metrics", "overall_acc"], None)  # may not exist
    swa = g(["metrics", "SWA"], None)
    cwa = g(["metrics", "CWA"], None)
    zs = g(["metrics", "ZSRTA"], [])
    zs = zs[-1] if isinstance(zs, list) and zs else None
    metrics = {
        k: v
        for k, v in zip(
            ["Overall Acc", "SWA", "CWA", "ZSRTA"], [overall_acc, swa, cwa, zs]
        )
        if v is not None
    }
    if metrics:
        plt.figure()
        plt.bar(
            range(len(metrics)), list(metrics.values()), tick_label=list(metrics.keys())
        )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Test Metrics")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

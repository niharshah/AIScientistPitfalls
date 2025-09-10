import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("uni_lstm", {}).get("spr", {})


# helper: convert list[(epoch,val)] -> np arrays
def to_xy(pairs):
    if not pairs:
        return np.array([]), np.array([])
    ep, val = zip(*pairs)
    return np.array(ep), np.array(val)


# ------------------------------------------------------------------
# 1) training / validation loss curves
try:
    train_e, train_l = to_xy(spr_data.get("losses", {}).get("train", []))
    val_e, val_l = to_xy(spr_data.get("losses", {}).get("val", []))

    if train_e.size and val_e.size:
        plt.figure()
        plt.plot(train_e, train_l, label="Train Loss")
        plt.plot(val_e, val_l, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_loss_curves.png")
        plt.savefig(fname)
    else:
        print("Loss data missing, skipping loss plot.")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) validation metric curves (CWA / SWA / PCWA)
try:
    val_metrics = spr_data.get("metrics", {}).get("val", [])
    if val_metrics:
        epochs = np.array([ep for ep, _ in val_metrics])
        cwa = np.array([m["CWA"] for _, m in val_metrics])
        swa = np.array([m["SWA"] for _, m in val_metrics])
        pcwa = np.array([m["PCWA"] for _, m in val_metrics])

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for ax, y, ttl in zip(axs, [cwa, swa, pcwa], ["CWA", "SWA", "PCWA"]):
            ax.plot(epochs, y)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ttl)
            ax.set_title(ttl)
        fig.suptitle("SPR Dataset – Validation Metrics")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, "spr_validation_metrics.png")
        plt.savefig(fname)
    else:
        print("Metric data missing, skipping metric plots.")
    plt.close("all")
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close("all")

# ------------------------------------------------------------------
# print test metrics
test_m = spr_data.get("metrics", {}).get("test", {})
if test_m:
    print("Test Metrics:", test_m)
else:
    print("No test metrics found.")

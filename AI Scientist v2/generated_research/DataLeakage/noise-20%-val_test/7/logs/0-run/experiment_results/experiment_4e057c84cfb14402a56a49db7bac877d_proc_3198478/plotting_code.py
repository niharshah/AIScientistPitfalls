import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("SPR_BENCH", {})
metrics = spr_data.get("metrics", {})
losses = spr_data.get("losses", {})


# Helper to fetch metric lists safely
def m(key):
    return metrics.get(key, [])


# ---------------------------  PLOTS  --------------------------------------
# 1) Accuracy curves
try:
    train_acc, val_acc = m("train_acc"), m("val_acc")
    if train_acc and val_acc:
        epochs = range(1, len(train_acc) + 1)
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Loss curves
try:
    train_loss, val_loss = losses.get("train", []), losses.get("val", [])
    if train_loss and val_loss:
        epochs = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Fidelity and FAGM curves
try:
    val_fid, val_fagm = m("val_fidelity"), m("val_fagm")
    if val_fid and val_fagm:
        epochs = range(1, len(val_fid) + 1)
        plt.figure()
        plt.plot(epochs, val_fid, label="Validation Fidelity")
        plt.plot(epochs, val_fagm, label="Validation FAGM")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("SPR_BENCH Explainability Metrics\nLeft: Fidelity, Right: FAGM")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_fidelity_fagm.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating fidelity/FAGM plot: {e}")
    plt.close()

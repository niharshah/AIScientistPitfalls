import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested keys
def _get(dic, *keys, default=None):
    for k in keys:
        dic = dic.get(k, {})
    return dic if dic else default


run = _get(experiment_data, "bag_of_embeddings", "SPR_BENCH", default={})
epochs = np.array(run.get("epochs", []))
train_loss = np.array(run.get("losses", {}).get("train", []))
val_loss = np.array(run.get("losses", {}).get("val", []))
val_metric = np.array(run.get("metrics", {}).get("val", []))

# --------------------------------------------------------------------------- #
# Plot 1: Training & Validation Loss
try:
    if epochs.size and train_loss.size and val_loss.size:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# --------------------------------------------------------------------------- #
# Plot 2: Validation Shape-Weighted-Accuracy
try:
    if epochs.size and val_metric.size:
        plt.figure()
        plt.plot(epochs, val_metric, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted-Accuracy")
        plt.title("SPR_BENCH Validation SWA over Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

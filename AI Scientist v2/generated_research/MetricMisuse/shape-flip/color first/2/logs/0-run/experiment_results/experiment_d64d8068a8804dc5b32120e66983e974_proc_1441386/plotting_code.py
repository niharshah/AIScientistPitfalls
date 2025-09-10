import matplotlib.pyplot as plt
import numpy as np
import os

# working directory setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_root = experiment_data.get("batch_size_tuning", {}).get("SPR_BENCH", {})

# ---- Plot 1: Loss curves -----------------------------------------------------
try:
    plt.figure()
    for bs, entry in data_root.items():
        epochs = range(1, len(entry["losses"]["train"]) + 1)
        train_losses = [l for _, l in entry["losses"]["train"]]
        val_losses = [l for _, l in entry["losses"]["val"]]
        plt.plot(epochs, train_losses, label=f"train bs={bs}", linestyle="--")
        plt.plot(epochs, val_losses, label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs. Validation Loss across Batch Sizes")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---- Plot 2: Validation DWA ---------------------------------------------------
try:
    plt.figure()
    for bs, entry in data_root.items():
        epochs = range(1, len(entry["metrics"]["val"]) + 1)
        dwa_vals = [m for _, m in entry["metrics"]["val"]]
        plt.plot(epochs, dwa_vals, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Dual-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation DWA across Batch Sizes")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_validation_dwa.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating DWA plot: {e}")
    plt.close()

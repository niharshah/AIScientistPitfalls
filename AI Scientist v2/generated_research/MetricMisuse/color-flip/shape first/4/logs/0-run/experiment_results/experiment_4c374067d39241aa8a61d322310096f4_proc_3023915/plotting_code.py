import matplotlib.pyplot as plt
import numpy as np
import os

# setup paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["epochs"]["SPR_BENCH"]  # dict keyed by epoch count
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# ---------- Plot 1: Train / Val loss curves ----------
try:
    if runs:
        plt.figure(figsize=(7, 5))
        for run_name, rec in runs.items():
            x = np.arange(1, len(rec["losses"]["train"]) + 1)
            plt.plot(
                x, rec["losses"]["train"], label=f"{run_name}ep-train", linestyle="--"
            )
            plt.plot(x, rec["losses"]["val"], label=f"{run_name}ep-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No runs to plot for loss curves.")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
finally:
    plt.close()

# ---------- Plot 2: Validation HWA curves ----------
try:
    if runs:
        plt.figure(figsize=(7, 5))
        for run_name, rec in runs.items():
            x = np.arange(1, len(rec["metrics"]["val"]) + 1)
            plt.plot(x, rec["metrics"]["val"], label=f"{run_name}ep")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("SPR_BENCH: Validation HWA Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No runs to plot for HWA curves.")
except Exception as e:
    print(f"Error creating HWA plot: {e}")
finally:
    plt.close()

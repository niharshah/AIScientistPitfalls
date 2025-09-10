import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
loss_train = loss_val = val_metrics = []
if ds_name in experiment_data:
    loss_train = experiment_data[ds_name]["losses"]["train"]
    loss_val = experiment_data[ds_name]["losses"]["val"]
    val_metrics = experiment_data[ds_name]["metrics"]["val"]

# ------------------------------------------------------------------
# Plot 1: Train & Val loss curves
try:
    if loss_train and loss_val:
        epochs = np.arange(1, len(loss_train) + 1)
        plt.figure()
        plt.plot(epochs, loss_train, label="Train Loss")
        plt.plot(epochs, loss_val, label="Val Loss")
        plt.title(f"{ds_name} – Loss Curves\nSolid: Train, Dashed: Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 2: Validation metrics (SWA, CWA, CompWA)
try:
    if val_metrics:
        swa = [m["SWA"] for m in val_metrics]
        cwa = [m["CWA"] for m in val_metrics]
        cpwa = [m["CompWA"] for m in val_metrics]
        epochs = np.arange(1, len(val_metrics) + 1)
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, cpwa, label="CompWA")
        plt.title(f"{ds_name} – Validation Weighted Accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_val_metrics.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final validation metrics
if val_metrics:
    final = val_metrics[-1]
    print(
        f"Final Val Metrics – SWA:{final['SWA']:.3f}  "
        f"CWA:{final['CWA']:.3f}  CompWA:{final['CompWA']:.3f}"
    )

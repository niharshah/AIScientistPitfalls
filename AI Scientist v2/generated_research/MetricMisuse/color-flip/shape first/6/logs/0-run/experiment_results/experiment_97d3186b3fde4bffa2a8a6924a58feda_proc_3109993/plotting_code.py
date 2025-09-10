import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

if experiment_data:
    spr = experiment_data["NoProjectionHead"]["SPR"]
    epochs = spr["epochs"]
    tr_loss = spr["losses"]["train"]
    val_loss = spr["losses"]["val"]
    swa = [m["SWA"] for m in spr["metrics"]["val"]]
    cwa = [m["CWA"] for m in spr["metrics"]["val"]]
    scaa = [m["SCAA"] for m in spr["metrics"]["val"]]

    # -----------------------------------------------------------------
    # Plot 1: Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -----------------------------------------------------------------
    # Plot 2: Validation metrics
    try:
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, scaa, label="SCAA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR Dataset: Validation Metrics")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_validation_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # -----------------------------------------------------------------
    print(
        "Final Validation Metrics:",
        spr["metrics"]["val"][-1] if spr["metrics"]["val"] else {},
    )
else:
    print("No experiment data found – nothing to plot.")

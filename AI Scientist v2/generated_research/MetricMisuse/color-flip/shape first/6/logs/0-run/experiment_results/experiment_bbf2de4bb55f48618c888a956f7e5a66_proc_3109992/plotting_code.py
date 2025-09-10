import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["identity_views"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data:
    epochs = spr_data["epochs"]
    tr_loss = spr_data["losses"]["train"]
    val_loss = spr_data["losses"]["val"]
    val_metrics = spr_data["metrics"]["val"]  # list of dicts
    swa = [m["SWA"] for m in val_metrics]
    cwa = [m["CWA"] for m in val_metrics]
    scaa = [m["SCAA"] for m in val_metrics]

    # --------------------------------------------------------------
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.title("SPR Dataset – Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved loss curves to {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------------------------------------------------------------
    # 2) Metric curves
    try:
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, scaa, label="SCAA")
        plt.title("SPR Dataset – Validation Metrics Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved validation metrics to {fname}")
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # --------------------------------------------------------------
    # Print final epoch metrics
    print("Final Validation Metrics:")
    print(f"  SWA  = {swa[-1]:.3f}")
    print(f"  CWA  = {cwa[-1]:.3f}")
    print(f"  SCAA = {scaa[-1]:.3f}")

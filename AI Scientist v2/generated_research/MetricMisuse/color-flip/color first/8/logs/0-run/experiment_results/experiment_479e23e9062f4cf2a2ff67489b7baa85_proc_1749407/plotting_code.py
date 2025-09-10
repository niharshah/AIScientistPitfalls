import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------- pick first (and only) dataset ----------------
try:
    run_key = next(iter(experiment_data.keys()))
    ds_key = next(iter(experiment_data[run_key].keys()))
    logs = experiment_data[run_key][ds_key]
except StopIteration:
    logs = None

if logs is None:
    print("No data found to plot.")
else:
    # -------- Figure 1 : loss curves --------
    try:
        train_loss = np.array(logs["losses"]["train"])
        val_loss = np.array(logs["losses"]["val"])
        plt.figure()
        plt.plot(train_loss[:, 0], train_loss[:, 1], label="Train loss")
        plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy")
        plt.title(f"{run_key} – {ds_key} Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_loss_curves_{run_key}.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------- Figure 2 : validation metrics (CWA, SWA, HM) --------
    try:
        val_metrics = np.array(logs["metrics"]["val"])
        epochs = val_metrics[:, 0]
        cwa = val_metrics[:, 1]
        swa = val_metrics[:, 2]
        hm = val_metrics[:, 3]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, hm, label="HM")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{run_key} – {ds_key} Validation Metrics")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_validation_metrics_{run_key}.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

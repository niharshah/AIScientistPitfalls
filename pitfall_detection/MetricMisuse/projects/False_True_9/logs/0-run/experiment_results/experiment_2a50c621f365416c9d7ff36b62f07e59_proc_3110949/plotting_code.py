import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load experiment results ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# --- iterate over datasets in the file ---
for ds_name, ds_dict in experiment_data.items():
    # ---------------- Loss curves ----------------
    try:
        # unpack losses
        ep_tr, tr_loss = zip(*ds_dict["losses"]["train"])
        ep_val, val_loss = zip(*ds_dict["losses"]["val"])
        plt.figure()
        plt.plot(ep_tr, tr_loss, marker="o", label="Train")
        plt.plot(ep_val, val_loss, marker="s", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curves\nTraining vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # -------------- Metric curves ----------------
    try:
        metrics = ds_dict["metrics"]["val"]  # [(ep, swa, cwa, dwa, hwa), ...]
        epochs, swa, cwa, dwa, hwa = zip(*metrics)
        plt.figure()
        plt.plot(epochs, swa, marker="o", label="SWA")
        plt.plot(epochs, cwa, marker="s", label="CWA")
        plt.plot(epochs, dwa, marker="^", label="DWA")
        plt.plot(epochs, hwa, marker="d", label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Weighted-Accuracy Curves\nSWA / CWA / DWA / HWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_weighted_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {ds_name}: {e}")
        plt.close()

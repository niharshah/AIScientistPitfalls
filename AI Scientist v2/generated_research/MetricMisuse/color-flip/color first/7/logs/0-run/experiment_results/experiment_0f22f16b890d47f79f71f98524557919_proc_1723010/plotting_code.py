import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------
for ds_name, ds_dict in experiment_data.items():
    epochs = ds_dict.get("epochs", [])
    train_losses = ds_dict.get("losses", {}).get("train", [])
    val_losses = ds_dict.get("losses", {}).get("val", [])
    train_metrics = ds_dict.get("metrics", {}).get("train", [])
    val_metrics = ds_dict.get("metrics", {}).get("val", [])

    # ------------ 1) Training loss curve -----------------------
    try:
        if train_losses:
            plt.figure()
            plt.plot(epochs, train_losses, marker="o", label="Train")
            plt.title(f"{ds_name} Dataset – Training Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name.lower()}_training_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating training-loss plot for {ds_name}: {e}")
        plt.close()

    # ------------ 2) Validation weighted-accuracy metrics ------
    try:
        if val_metrics:
            cwa = [m["cwa"] for m in val_metrics]
            swa = [m["swa"] for m in val_metrics]
            cpx = [m["cpx"] for m in val_metrics]
            plt.figure()
            plt.plot(epochs, cwa, marker="o", label="CWA")
            plt.plot(epochs, swa, marker="s", label="SWA")
            plt.plot(epochs, cpx, marker="^", label="CpxWA")
            plt.title(f"{ds_name} Dataset – Validation Weighted-Accuracy Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name.lower()}_val_weighted_acc.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating validation-metrics plot for {ds_name}: {e}")
        plt.close()

    # ------------ 3) Train vs Val CpxWA ------------------------
    try:
        if train_metrics and val_metrics:
            train_cpx = [m["cpx"] for m in train_metrics]
            val_cpx = [m["cpx"] for m in val_metrics]
            plt.figure()
            plt.plot(epochs, train_cpx, marker="o", label="Train CpxWA")
            plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
            plt.title(f"{ds_name} Dataset – CpxWA Train vs. Val")
            plt.xlabel("Epoch")
            plt.ylabel("CpxWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name.lower()}_cpxwa_train_val.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating CpxWA comparison plot for {ds_name}: {e}")
        plt.close()

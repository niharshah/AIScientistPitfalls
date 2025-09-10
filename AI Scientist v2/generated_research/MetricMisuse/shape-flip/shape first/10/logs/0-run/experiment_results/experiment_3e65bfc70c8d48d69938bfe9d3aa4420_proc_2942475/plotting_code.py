import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dname, record in experiment_data.items():
    losses = record.get("losses", {})
    metrics = record.get("metrics", {})
    # -------- Plot 1: loss curves --------
    try:
        plt.figure()
        if "train" in losses and losses["train"]:
            plt.plot(losses["train"], label="Train Loss")
        if "val" in losses and losses["val"]:
            plt.plot(losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname} Training vs Validation Loss")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # -------- Plot 2: validation metrics --------
    try:
        plt.figure()
        val_metrics = metrics.get("val", [])
        if val_metrics:
            epochs = range(len(val_metrics))
            crwa = [m["CRWA"] for m in val_metrics]
            swa = [m["SWA"] for m in val_metrics]
            cwa = [m["CWA"] for m in val_metrics]
            plt.plot(epochs, crwa, label="CRWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cwa, label="CWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dname} Validation Metrics over Epochs")
            plt.legend()
            save_path = os.path.join(working_dir, f"{dname}_val_metric_curves.png")
            plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dname}: {e}")
        plt.close()

    # -------- Plot 3: test metrics bar --------
    try:
        test_m = metrics.get("test", {})
        if test_m:
            plt.figure()
            names = list(test_m.keys())
            vals = [test_m[k] for k in names]
            plt.bar(names, vals)
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(f"{dname} Final Test Metrics")
            save_path = os.path.join(working_dir, f"{dname}_test_metrics_bar.png")
            plt.savefig(save_path)
            print(f"{dname} test metrics:", test_m)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar for {dname}: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Iterate through stored results
for model_name, datasets in experiment_data.items():
    for dset_name, ed in datasets.items():
        epochs = ed.get("epochs", [])
        train_loss = ed.get("losses", {}).get("train", [])
        val_loss = ed.get("losses", {}).get("val", [])
        val_metrics = ed.get("metrics", {}).get("val", [])

        # 1) Loss curves -----------------------------------------------------
        try:
            plt.figure()
            if train_loss:
                plt.plot(epochs, train_loss, label="Train Loss")
            if val_loss:
                plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name} Loss Curves ({model_name})\nTraining vs Validation")
            plt.legend()
            fname = f"{dset_name}_loss_curve_{model_name}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset_name}: {e}")
            plt.close()

        # 2) Validation metric curves ----------------------------------------
        try:
            if val_metrics:
                swa = [m["SWA"] for m in val_metrics]
                cwa = [m["CWA"] for m in val_metrics]
                scaa = [m["SCAA"] for m in val_metrics]

                plt.figure()
                plt.plot(epochs, swa, label="SWA")
                plt.plot(epochs, cwa, label="CWA")
                plt.plot(epochs, scaa, label="SCAA")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(
                    f"{dset_name} Validation Metrics ({model_name})\nSWA, CWA, SCAA"
                )
                plt.legend()
                fname = f"{dset_name}_val_metrics_{model_name}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()

                print(f"Final {dset_name} metrics ({model_name}): {val_metrics[-1]}")
        except Exception as e:
            print(f"Error creating metric plot for {dset_name}: {e}")
            plt.close()

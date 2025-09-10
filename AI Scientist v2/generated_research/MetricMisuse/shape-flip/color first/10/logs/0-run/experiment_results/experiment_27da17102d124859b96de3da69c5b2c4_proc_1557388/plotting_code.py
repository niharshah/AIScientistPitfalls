import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----- iterate and plot -----
for ablation, ds_dict in experiment_data.items():
    for dset, info in ds_dict.items():
        train_losses = info.get("losses", {}).get("train", [])
        val_losses = info.get("losses", {}).get("val", [])
        val_metrics = info.get("metrics", {}).get("val", [])

        epochs = range(1, len(train_losses) + 1)
        val_epochs = [m["epoch"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        swa = [m["swa"] for m in val_metrics]
        cpx = [m["cpxwa"] for m in val_metrics]

        # 1) loss curve --------------------------------------------------------
        try:
            if train_losses and val_losses:
                plt.figure()
                plt.plot(epochs, train_losses, label="Train Loss")
                plt.plot(epochs, val_losses, label="Val Loss")
                plt.title(f"{dset} Loss Curve ({ablation})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(
                    os.path.join(working_dir, f"{dset}_{ablation}_loss_curve.png")
                )
                plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dset}: {e}")
            plt.close()

        # 2) validation weighted-accuracy curves ------------------------------
        try:
            if val_epochs:
                plt.figure()
                plt.plot(val_epochs, cwa, label="CWA")
                plt.plot(val_epochs, swa, label="SWA")
                plt.plot(val_epochs, cpx, label="CpxWA")
                plt.title(f"{dset} Validation Weighted Accuracies ({ablation})")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{dset}_{ablation}_val_weighted_accuracies.png"
                    )
                )
                plt.close()
        except Exception as e:
            print(f"Error creating val accuracy curve for {dset}: {e}")
            plt.close()

        # 3) test weighted-accuracy bar chart ---------------------------------
        try:
            test_m = info.get("metrics", {}).get("test", {})
            if test_m:
                labels = ["CWA", "SWA", "CpxWA"]
                values = [
                    test_m.get("cwa", 0),
                    test_m.get("swa", 0),
                    test_m.get("cpxwa", 0),
                ]
                plt.figure()
                plt.bar(labels, values)
                plt.ylim(0, 1)
                plt.title(f"{dset} Test Weighted Accuracies ({ablation})")
                plt.savefig(
                    os.path.join(
                        working_dir, f"{dset}_{ablation}_test_weighted_accuracies.png"
                    )
                )
                plt.close()
        except Exception as e:
            print(f"Error creating test accuracy bar for {dset}: {e}")
            plt.close()

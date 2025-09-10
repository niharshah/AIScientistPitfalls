import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- Setup -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

fig_cnt, FIG_LIMIT = 0, 5

for branch, dsets in experiment_data.items():
    for dset_name, content in dsets.items():
        losses = content.get("losses", {})
        metrics = content.get("metrics", {})
        # -------- Loss curve ------------------------------------------
        if fig_cnt < FIG_LIMIT and losses.get("train") and losses.get("val"):
            try:
                plt.figure()
                epochs = range(1, len(losses["train"]) + 1)
                plt.plot(epochs, losses["train"], label="Train")
                plt.plot(epochs, losses["val"], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{branch} on {dset_name} – Loss Curves")
                plt.legend()
                fname = f"{branch}_{dset_name}_loss_curves.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
                fig_cnt += 1
            except Exception as e:
                print(f"Error creating loss plot for {dset_name}: {e}")
                plt.close()
        # -------- Metric curve ----------------------------------------
        if fig_cnt < FIG_LIMIT and metrics.get("val"):
            try:
                plt.figure()
                epochs = range(1, len(metrics["val"]) + 1)
                if metrics.get("train"):
                    plt.plot(epochs, metrics["train"], label="Train", linestyle="--")
                plt.plot(epochs, metrics["val"], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Shape-Weighted Accuracy")
                plt.title(f"{branch} on {dset_name} – Accuracy Curves")
                plt.legend()
                fname = f"{branch}_{dset_name}_accuracy_curves.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
                fig_cnt += 1
            except Exception as e:
                print(f"Error creating accuracy plot for {dset_name}: {e}")
                plt.close()
        # -------- Print test metric ----------------------------------
        if "test" in metrics:
            print(f'Test SWA for {branch}/{dset_name}: {metrics["test"]:.4f}')

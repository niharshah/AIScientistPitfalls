import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------------- plotting helper ------------------
def save_plot(fig, name):
    fig.savefig(os.path.join(working_dir, name))
    plt.close(fig)


# -------------- iterate over experiments -----------
for exp_name, exp_val in experiment_data.items():
    for dset_name, dset_val in exp_val.items():
        epochs = dset_val.get("epochs", [])
        losses = dset_val.get("losses", {})
        metrics = dset_val.get("metrics", {})
        # -------- 1. loss curve -------------
        try:
            fig = plt.figure()
            plt.plot(epochs, losses["train"], label="Train")
            plt.plot(epochs, losses["val"], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name} Loss Curve\nTrain vs Val")
            plt.legend()
            save_plot(fig, f"{dset_name}_loss_curve.png")
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # -------- 2. CWA curve --------------
        try:
            fig = plt.figure()
            plt.plot(epochs, metrics["train"]["CWA"], label="Train")
            plt.plot(epochs, metrics["val"]["CWA"], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("CWA")
            plt.title(f"{dset_name} Color-Weighted Accuracy\nTrain vs Val")
            plt.legend()
            save_plot(fig, f"{dset_name}_CWA_curve.png")
        except Exception as e:
            print(f"Error creating CWA plot: {e}")
            plt.close()

        # -------- 3. SWA curve --------------
        try:
            fig = plt.figure()
            plt.plot(epochs, metrics["train"]["SWA"], label="Train")
            plt.plot(epochs, metrics["val"]["SWA"], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.title(f"{dset_name} Shape-Weighted Accuracy\nTrain vs Val")
            plt.legend()
            save_plot(fig, f"{dset_name}_SWA_curve.png")
        except Exception as e:
            print(f"Error creating SWA plot: {e}")
            plt.close()

        # -------- 4. CplxWA curve -----------
        try:
            fig = plt.figure()
            plt.plot(epochs, metrics["train"]["CplxWA"], label="Train")
            plt.plot(epochs, metrics["val"]["CplxWA"], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("CplxWA")
            plt.title(f"{dset_name} Complexity-Weighted Accuracy\nTrain vs Val")
            plt.legend()
            save_plot(fig, f"{dset_name}_CplxWA_curve.png")
        except Exception as e:
            print(f"Error creating CplxWA plot: {e}")
            plt.close()

        # -------- print test metrics --------
        test_metrics = metrics.get("test", {})
        if test_metrics:
            print(
                f"{dset_name} Test Metrics — "
                f"CWA: {test_metrics.get('CWA', 'NA'):.3f}, "
                f"SWA: {test_metrics.get('SWA', 'NA'):.3f}, "
                f"CplxWA: {test_metrics.get('CplxWA', 'NA'):.3f}"
            )

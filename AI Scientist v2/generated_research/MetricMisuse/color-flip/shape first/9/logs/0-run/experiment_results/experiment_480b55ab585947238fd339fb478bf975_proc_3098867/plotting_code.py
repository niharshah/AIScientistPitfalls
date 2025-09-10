import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper: gather per-tag arrays
tags = sorted(experiment_data.keys())
train_losses, val_losses, val_cwa = {}, {}, {}
for tag in tags:
    try:
        d = experiment_data[tag]["SPR_BENCH"]
        train_losses[tag] = d["losses"]["train"]
        val_losses[tag] = d["losses"]["val"]
        val_cwa[tag] = d["metrics"]["val"]
    except Exception:
        continue  # skip incomplete tags


# -------------------- plot functions --------------------
def plot_metric(metric_dict, ylabel, figname):
    try:
        plt.figure()
        for tag, arr in metric_dict.items():
            plt.plot(range(1, len(arr) + 1), arr, label=tag)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"SPR_BENCH {ylabel} curves")
        plt.legend()
        save_path = os.path.join(working_dir, f"SPR_BENCH_{figname}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating {figname} plot: {e}")
        plt.close()


# -------------------- create plots --------------------
plot_metric(train_losses, "Train Loss", "train_loss")
plot_metric(val_losses, "Validation Loss", "val_loss")
plot_metric(val_cwa, "Validation CWA-2D", "val_cwa")

print("Plots saved to", working_dir)

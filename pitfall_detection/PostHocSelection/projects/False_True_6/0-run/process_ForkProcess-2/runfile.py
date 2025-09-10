import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load experiment data -------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to limit epochs plotted
def epoch_subsample(x, max_points=5):
    if len(x) <= max_points:
        return np.arange(1, len(x) + 1), x
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return idx + 1, [x[i] for i in idx]


for dset, data in experiment_data.items():
    # ---------- 1. Loss curves ----------
    try:
        train_losses = data["losses"].get("train", [])
        val_losses = data["losses"].get("val", [])
        if train_losses and val_losses:
            ep_t, train_plot = epoch_subsample(train_losses)
            ep_v, val_plot = epoch_subsample(val_losses)
            plt.figure()
            plt.plot(ep_t, train_plot, label="Train Loss")
            plt.plot(ep_v, val_plot, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset} Loss Curves\nTrain vs. Validation")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset.lower()}_loss_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ---------- 2. Validation metric trends ----------
    try:
        val_metrics = data["metrics"].get("val", [])
        if val_metrics:
            epochs = [m.get("epoch", i + 1) for i, m in enumerate(val_metrics)]
            metrics_to_plot = {
                k: [m.get(k) for m in val_metrics]
                for k in ("acc", "swa", "cwa", "nrgs")
                if val_metrics[0].get(k) is not None
            }
            plt.figure()
            for name, values in metrics_to_plot.items():
                ep_s, vals_s = epoch_subsample(values)
                plt.plot(ep_s, vals_s, label=name.upper())
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.ylim(0, 1)
            plt.title(f"{dset} Validation Metrics Across Epochs")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset.lower()}_val_metrics.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dset}: {e}")
        plt.close()

    # ---------- 3. Confusion matrix ----------
    try:
        preds = np.array(data.get("predictions", []))
        trues = np.array(data.get("ground_truth", []))
        if preds.size and trues.size:
            labels = np.unique(np.concatenate([preds, trues]))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(trues, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dset} Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.xticks(labels)
            plt.yticks(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset.lower()}_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

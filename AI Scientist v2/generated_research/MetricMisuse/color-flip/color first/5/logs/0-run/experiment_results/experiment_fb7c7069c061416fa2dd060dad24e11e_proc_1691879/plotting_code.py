import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns  # lightweight; if unavailable we fallback to plt.imshow

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ablation_key = next(iter(experiment_data))
    ds_key = next(iter(experiment_data[ablation_key]))
    ed = experiment_data[ablation_key][ds_key]

    # helper
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # ------------- Plot 1: loss curves -------------
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_key} Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------- Plot 2: validation metrics -------------
    try:
        metrics_per_epoch = ed["metrics"]["val"]
        if metrics_per_epoch:
            acc = [m["acc"] for m in metrics_per_epoch]
            cwa = [m["cwa"] for m in metrics_per_epoch]
            swa = [m["swa"] for m in metrics_per_epoch]
            ccwa = [m["ccwa"] for m in metrics_per_epoch]
            plt.figure()
            plt.plot(epochs, acc, label="ACC")
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, ccwa, label="CCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(
                f"{ds_key} Validation Metrics Over Epochs\nLeft: ACC, Right: Weighted Accuracies"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_key}_validation_metrics.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # ------------- Plot 3: confusion matrix -------------
    try:
        y_true = np.array(ed["ground_truth"])
        y_pred = np.array(ed["predictions"])
        if y_true.size and y_pred.size:
            cm = confusion_matrix(y_true, y_pred, normalize="true")
            plt.figure()
            try:
                sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
            except Exception:
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{ds_key} Test Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            fname = os.path.join(working_dir, f"{ds_key}_confusion_matrix.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

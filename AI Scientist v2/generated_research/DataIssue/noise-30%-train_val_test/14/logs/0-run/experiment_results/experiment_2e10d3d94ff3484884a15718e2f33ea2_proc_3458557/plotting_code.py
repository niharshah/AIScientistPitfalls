import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------------------- iterate over datasets -------------------
for dset_name, dset_data in experiment_data.items():
    # ---------- collect per-epoch stats ----------
    train_losses = [d["loss"] for d in dset_data["losses"]["train"]]
    val_losses = [d["loss"] for d in dset_data["losses"]["val"]]
    train_f1s = [d["macro_f1"] for d in dset_data["metrics"]["train"]]
    val_f1s = [d["macro_f1"] for d in dset_data["metrics"]["val"]]
    epochs = [d["epoch"] for d in dset_data["metrics"]["train"]]

    # ---------- plot: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{dset_name} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # ---------- plot: macro-F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_f1s, label="Train Macro-F1")
        plt.plot(epochs, val_f1s, label="Validation Macro-F1")
        plt.title(f"{dset_name} Macro-F1 Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset_name}_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {dset_name}: {e}")
        plt.close()

    # ---------- plot: confusion matrix (dev set) ----------
    try:
        y_true = dset_data["ground_truth"]
        y_pred = dset_data["predictions"]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(
            f"{dset_name} Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
        )
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

    # ---------- print final val macro-F1 ----------
    if val_f1s:
        print(f"{dset_name} final validation Macro-F1: {val_f1s[-1]:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# iterate over available datasets
for dset, logs in experiment_data.items():
    # ----- 1. Loss curves -----
    try:
        epochs = range(1, len(logs["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, logs["losses"]["train"], label="Train Loss")
        plt.plot(epochs, logs["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset} – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ----- 2. Macro-F1 curves -----
    try:
        epochs = range(1, len(logs["metrics"]["train_macro_f1"]) + 1)
        plt.figure()
        plt.plot(epochs, logs["metrics"]["train_macro_f1"], label="Train Macro-F1")
        plt.plot(epochs, logs["metrics"]["val_macro_f1"], label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dset} – Training vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {dset}: {e}")
        plt.close()

    # ----- 3. Confusion matrix on test set -----
    try:
        y_true = np.array(logs["ground_truth"])
        y_pred = np.array(logs["predictions"])
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"{dset} – Confusion Matrix\nRows: Ground Truth, Cols: Predictions")
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()
print(f"Plots saved to {working_dir}")

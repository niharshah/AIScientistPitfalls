import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ----------------- helper for confusion matrix -----------------
def confusion(y_true, y_pred, num_cls):
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ----------------- plot for every dataset -----------------
for dset, content in experiment_data.items():
    # common handles
    epochs = np.arange(1, len(content["losses"]["train"]) + 1)
    # 1) Loss curve -------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, content["losses"]["train"], label="train")
        plt.plot(epochs, content["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # 2) Macro-F1 curve --------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, content["metrics"]["train_macroF1"], label="train")
        plt.plot(epochs, content["metrics"]["val_macroF1"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dset}: Training vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_macroF1_curve.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating F1 curve for {dset}: {e}")
        plt.close()

    # 3) Confusion matrix ------------------------------------------
    try:
        y_true = np.array(content["ground_truth"])
        y_pred = np.array(content["predictions"])
        num_cls = int(max(y_true.max(), y_pred.max()) + 1) if y_true.size else 0
        if num_cls:
            cm = confusion(y_true, y_pred, num_cls)
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset}: Confusion Matrix (Validation)")
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print("Saved", fname)
        else:
            print(f"No data for confusion matrix plot for {dset}")
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

    # 4) Class count bar chart -------------------------------------
    try:
        if y_true.size:
            labels = np.arange(num_cls)
            true_counts = np.bincount(y_true, minlength=num_cls)
            pred_counts = np.bincount(y_pred, minlength=num_cls)
            x = np.arange(num_cls)
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, true_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pred_counts, width, label="Predictions")
            plt.xlabel("Class id")
            plt.ylabel("Count")
            plt.title(f"{dset}: Class Distribution (Validation)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_class_distribution.png")
            plt.savefig(fname)
            plt.close()
            print("Saved", fname)
        else:
            print(f"No data for class distribution plot for {dset}")
    except Exception as e:
        print(f"Error creating class distribution plot for {dset}: {e}")
        plt.close()

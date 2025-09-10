import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment results -----------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Utility: simple confusion matrix ---------------------------------------------------------
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# Plotting ---------------------------------------------------------------------------------
for dset_name, logs in experiment_data.items():
    epochs = logs.get("epochs", [])
    losses_tr = logs.get("losses", {}).get("train", [])
    losses_val = logs.get("losses", {}).get("val", [])
    acc_tr = logs.get("metrics", {}).get("train", [])
    acc_val = logs.get("metrics", {}).get("val", [])
    y_pred = logs.get("predictions", [])
    y_true = logs.get("ground_truth", [])
    n_classes = max(y_true) + 1 if y_true else 0

    # 1) Loss curve ------------------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset_name}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve for {dset_name}: {e}")
        plt.close()

    # 2) Accuracy curve --------------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, acc_tr, label="Train")
        plt.plot(epochs, acc_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"{dset_name}: Training vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset_name}_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating accuracy curve for {dset_name}: {e}")
        plt.close()

    # 3) Confusion matrix ------------------------------------------------------------------
    try:
        if y_true and y_pred:
            cm = confusion_matrix(y_true, y_pred, n_classes)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset_name}: Test Confusion Matrix")
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

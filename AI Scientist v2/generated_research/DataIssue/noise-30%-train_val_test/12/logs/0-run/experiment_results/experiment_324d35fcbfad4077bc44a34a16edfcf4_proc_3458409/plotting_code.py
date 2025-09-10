import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load stored experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
for dset_name, d in experiment_data.items():
    epochs = d.get("epochs", [])
    # 1) Loss curve -------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, d["losses"]["train"], label="Train")
        plt.plot(epochs, d["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name} Loss Curve")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset_name}: {e}")
        plt.close()

    # 2) Macro-F1 curve --------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, d["metrics"]["train_f1"], label="Train")
        plt.plot(epochs, d["metrics"]["val_f1"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dset_name} Macro-F1 Curve")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dset_name}_f1_curve.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve for {dset_name}: {e}")
        plt.close()

    # 3) Confusion matrix ------------------------------------------
    try:
        y_true = d["ground_truth"]
        y_pred = d["predictions"]
        if len(y_true) and len(y_true) == len(y_pred):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset_name} Test Confusion Matrix")
            # annotate cells
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8,
                    )
            plt.tight_layout()
            save_path = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- load data ---------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------------- helper ------------------
def confusion_counts(y_true, y_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[labels.index(t)][labels.index(p)] += 1
    return cm


# -------------- iterate over datasets ---------------
for ds_name, ds_blob in experiment_data.items():
    # unpack lists
    tr_loss = ds_blob["losses"]["train"]
    va_loss = ds_blob["losses"]["val"]
    tr_met = ds_blob["metrics"]["train"]
    va_met = ds_blob["metrics"]["val"]

    epochs = [e for e, _ in tr_loss]
    tr_loss_vals = [v for _, v in tr_loss]
    va_loss_vals = [v for _, v in va_loss]
    tr_swa_vals = [v for _, v in tr_met]
    va_swa_vals = [v for _, v in va_met]

    # --------- Figure 1 : Loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_loss_vals, label="Train")
        plt.plot(epochs, va_loss_vals, label="Validation")
        plt.title(f"{ds_name} – Cross-Entropy Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_loss_curves.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds_name}: {e}")
        plt.close()

    # --------- Figure 2 : SWA curves ----------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_swa_vals, label="Train")
        plt.plot(epochs, va_swa_vals, label="Validation")
        plt.title(f"{ds_name} – Shape-Weighted Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_swa_curves.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curves for {ds_name}: {e}")
        plt.close()

    # --------- Figure 3 : Confusion Matrix ----------
    gt = ds_blob.get("ground_truth", [])
    pr = ds_blob.get("predictions", [])
    if len(gt) and len(pr):
        try:
            labels = sorted(list(set(gt) | set(pr)))
            cm = confusion_counts(gt, pr, labels)
            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
            for i, j in product(range(len(labels)), repeat=2):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
                )
            plt.title(f"{ds_name} – Confusion Matrix (Test)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            fname = os.path.join(working_dir, f"{ds_name.lower()}_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()

    # --------- Evaluation metric ----------
    if len(gt):
        acc = np.mean(np.array(gt) == np.array(pr))
        print(f"{ds_name} – Test Accuracy: {acc:.4f}")

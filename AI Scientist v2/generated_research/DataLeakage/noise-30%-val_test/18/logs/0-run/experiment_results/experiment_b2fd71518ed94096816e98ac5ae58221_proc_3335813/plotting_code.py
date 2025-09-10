import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate over datasets ----------
for dset, data in experiment_data.items():
    epochs = data.get("epochs", [])
    tr_loss = data.get("losses", {}).get("train", [])
    val_loss = data.get("losses", {}).get("val", [])
    tr_mcc = data.get("metrics", {}).get("train_MCC", [])
    val_mcc = data.get("metrics", {}).get("val_MCC", [])
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # ---- 1. loss curves ----
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="train")
        plt.plot(epochs, val_loss, linestyle="--", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dset}: {e}")
        plt.close()

    # ---- 2. MCC curves ----
    try:
        if tr_mcc and val_mcc:
            plt.figure()
            plt.plot(epochs, tr_mcc, label="train")
            plt.plot(epochs, val_mcc, linestyle="--", label="validation")
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.title(f"{dset}: Training vs Validation MCC")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_MCC_curves.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating MCC curves for {dset}: {e}")
        plt.close()

    # ---- 3. confusion matrix ----
    try:
        if preds.size and gts.size:
            cm = np.zeros((2, 2), dtype=int)
            for g, p in zip(gts, preds):
                cm[g, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.title(f"{dset} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.colorbar()
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

    # ---- print final metric ----
    if preds.size and gts.size:
        test_mcc = matthews_corrcoef(gts, preds)
        print(f"{dset} – Final Test MCC: {test_mcc:.4f}")

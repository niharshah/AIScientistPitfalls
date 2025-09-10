import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

# ---------------- setup -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data is not None:
    epochs = list(range(1, len(data["losses"]["train"]) + 1))

    # ------------- 1. Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Train")
        plt.plot(epochs, data["losses"]["val"], label="Validation", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ------------- 2. MCC curves -----------
    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["train_MCC"], label="Train MCC")
        plt.plot(
            epochs, data["metrics"]["val_MCC"], label="Validation MCC", linestyle="--"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Matthews Corrcoef")
        plt.title("SPR_BENCH: Training vs Validation MCC")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curves: {e}")
        plt.close()

    # ----------- 3. Confusion matrix -------
    try:
        preds = np.array(data["predictions"])
        gts = np.array(data["ground_truth"])
        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.colorbar()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ----------- 4. Print final metrics ----
    test_mcc = matthews_corrcoef(gts, preds)
    test_f1 = f1_score(gts, preds, average="macro")
    print(f"Final Test MCC: {test_mcc:.4f} | Final Test macro-F1: {test_f1:.4f}")

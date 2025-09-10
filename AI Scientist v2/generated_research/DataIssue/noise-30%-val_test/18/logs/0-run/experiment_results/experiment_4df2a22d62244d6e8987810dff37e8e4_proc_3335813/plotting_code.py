import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up -----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data is not None:
    epochs = np.arange(1, len(data["losses"]["train"]) + 1)

    # ---------- 1. Loss curves -----------------
    try:
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Train")
        plt.plot(epochs, data["losses"]["val"], linestyle="--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------- 2. MCC curves ------------------
    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["train_MCC"], label="Train")
        plt.plot(epochs, data["metrics"]["val_MCC"], linestyle="--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews Corr. Coef.")
        plt.title("SPR_BENCH: Training vs Validation MCC")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_MCC_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curves: {e}")
        plt.close()

    # ---------- 3. Confusion matrix ------------
    try:
        preds = np.array(data["predictions"])
        gts = np.array(data["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- 4. Print metrics ---------------
    tp, tn = cm[1, 1], cm[0, 0]
    fp, fn = cm[0, 1], cm[1, 0]
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-9
    test_mcc = ((tp * tn) - (fp * fn)) / denom
    best_val_mcc = max(data["metrics"]["val_MCC"])
    print(f"Best Validation MCC = {best_val_mcc:.4f} | Test MCC = {test_mcc:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds = "SPR_BENCH"
if ds in experiment_data:
    data = experiment_data[ds]
    epochs = data["epochs"]
    tr_loss = data["losses"]["train"]
    va_loss = data["losses"]["val"]
    tr_f1 = data["metrics"]["train_macro_f1"]
    va_f1 = data["metrics"]["val_macro_f1"]
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # ---------- 1. loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2. macro-F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, va_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # ---------- 3. confusion matrix ----------
    try:
        if preds.size and gts.size:
            cm = confusion_matrix(gts, preds)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(values_format="d", cmap="Blues")
            plt.title("SPR_BENCH Confusion Matrix (Test Set)")
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- print evaluation metric ----------
    if preds.size and gts.size:
        test_macro_f1 = f1_score(gts, preds, average="macro")
        print(f"Test macro-F1 (recomputed) = {test_macro_f1:.4f}")

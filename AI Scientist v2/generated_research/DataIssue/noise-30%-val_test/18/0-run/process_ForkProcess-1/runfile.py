import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# ensure working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("SPR_BENCH", {})
epochs = np.array(data.get("epochs", []))
train_ls = np.array(data.get("losses", {}).get("train", []))
val_ls = np.array(data.get("losses", {}).get("val", []))
train_f1 = np.array(data.get("metrics", {}).get("train", []))
val_f1 = np.array(data.get("metrics", {}).get("val", []))
preds = np.array(data.get("predictions", []))
gts = np.array(data.get("ground_truth", []))

# print best dev F1 if available
if len(val_f1):
    print(f"Best Dev Macro-F1 (stored) = {val_f1.max():.4f}")

# -------- 1) Loss curve ------------
try:
    if len(epochs) and len(train_ls) and len(val_ls):
        plt.figure()
        plt.plot(epochs, train_ls, label="Train Loss")
        plt.plot(epochs, val_ls, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------- 2) F1 curve ---------------
try:
    if len(epochs) and len(train_f1) and len(val_f1):
        plt.figure()
        plt.plot(epochs, train_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# -------- 3) Confusion matrix -------
try:
    if len(preds) and len(gts):
        cm = confusion_matrix(gts, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        plt.figure()
        disp.plot(cmap="Blues", values_format="d")
        plt.title("SPR_BENCH: Confusion Matrix (Best Dev)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

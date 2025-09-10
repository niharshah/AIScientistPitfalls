import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ---------- paths ----------
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

ed = experiment_data.get("no_positional_encoding", {}).get("SPR_BENCH", {})

epochs = np.array(ed.get("epochs", []))
train_loss = np.array(ed.get("losses", {}).get("train", []))
val_loss = np.array(ed.get("losses", {}).get("val", []))
train_f1 = np.array(ed.get("metrics", {}).get("train", []))
val_f1 = np.array(ed.get("metrics", {}).get("val", []))
test_f1 = ed.get("metrics", {}).get("test", None)
SGA = ed.get("metrics", {}).get("SGA", None)
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# ---------- 1. Loss curve ----------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curve\nLeft: Train, Right: Validation")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- 2. F1 curve ----------
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train Macro-F1")
    plt.plot(epochs, val_f1, label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curve\nLeft: Train, Right: Validation")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_F1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# ---------- 3. Confusion matrix ----------
try:
    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix (Test)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print metrics ----------
if len(val_f1):
    print(f"Best validation macro-F1: {max(val_f1):.4f}")
if test_f1 is not None:
    print(f"Test macro-F1: {test_f1:.4f}")
if SGA is not None:
    print(f"Systematic Generalization Accuracy: {SGA:.4f}")

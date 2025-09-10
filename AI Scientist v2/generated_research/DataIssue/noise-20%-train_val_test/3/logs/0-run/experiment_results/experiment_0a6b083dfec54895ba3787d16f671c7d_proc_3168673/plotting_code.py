import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix
import itertools

# set up output directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["no_dropout"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = np.array(ed["epochs"])
train_loss = np.array(ed["metrics"]["train_loss"])
val_loss = np.array(ed["metrics"]["val_loss"])
val_f1 = np.array(ed["metrics"]["val_f1"])
y_pred = np.array(ed["predictions"])
y_true = np.array(ed["ground_truth"])

# compute and print test macro-F1
test_f1 = f1_score(y_true, y_pred, average="macro")
print(f"Test macro-F1: {test_f1:.4f}")

# 1) Train / Val loss curve
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Validation F1 curve
try:
    plt.figure()
    plt.plot(epochs, val_f1, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Validation Macro-F1 Across Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3) Confusion matrix heatmap (optional third plot)
try:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # annotate cells
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

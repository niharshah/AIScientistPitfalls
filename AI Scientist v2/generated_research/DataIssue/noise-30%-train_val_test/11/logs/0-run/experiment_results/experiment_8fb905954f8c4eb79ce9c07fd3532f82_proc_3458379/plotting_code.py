import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

spr_data = experiment_data.get("SPR_BENCH", {})
epochs = spr_data.get("epochs", [])
train_loss = spr_data.get("losses", {}).get("train", [])
val_loss = spr_data.get("losses", {}).get("val", [])
train_f1 = spr_data.get("metrics", {}).get("train_f1", [])
val_f1 = spr_data.get("metrics", {}).get("val_f1", [])
preds = spr_data.get("predictions", [])
gts = spr_data.get("ground_truth", [])

# -----------------------------------------------------------
# 1) Loss curve
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curve")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -----------------------------------------------------------
# 2) F1 curve
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train Macro F1")
    plt.plot(epochs, val_f1, label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Macro-F1 Curve")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# -----------------------------------------------------------
# 3) Confusion matrix (test set)
try:
    cm = confusion_matrix(gts, preds)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(
        "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -----------------------------------------------------------
# Print evaluation metric
if preds and gts:
    test_macro_f1 = f1_score(gts, preds, average="macro")
    print(f"Test Macro F1: {test_macro_f1:.4f}")

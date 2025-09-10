import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
exp_key = "Remove_Positional_Encoding"
ds_key = "SPR_BENCH"
ed = experiment_data.get(exp_key, {}).get(ds_key, {})

# Basic safety checks
epochs = ed.get("epochs", [])
train_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
train_f1 = ed.get("metrics", {}).get("train", [])
val_f1 = ed.get("metrics", {}).get("val", [])
y_pred = ed.get("predictions", [])
y_true = ed.get("ground_truth", [])
test_loss = ed.get("test_loss", None)
test_macroF1 = ed.get("test_macroF1", None)

# ------------------------------------------------------------------
# 1) Loss curve
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH – Loss Curves\nModel: Remove_Positional_Encoding")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) F1 curve
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH – Macro-F1 Curves\nModel: Remove_Positional_Encoding")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion matrix on test set
if len(y_true) and len(y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("SPR_BENCH – Confusion Matrix\nModel: Remove_Positional_Encoding")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

# ------------------------------------------------------------------
# Print stored evaluation numbers
print(f"Test loss   : {test_loss}")
print(f"Test macroF1: {test_macroF1}")
print("Plots saved to:", working_dir)

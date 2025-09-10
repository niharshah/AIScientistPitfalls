import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

# ------------------------------- #
# Paths and data loading
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("SPR_BENCH", {})
metrics = data.get("metrics", {})
losses = data.get("losses", {})
epochs = data.get("epochs", [])

# ------------------------------- #
# Plot 1: Loss curves
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train Loss")
    plt.plot(epochs, losses.get("val", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------- #
# Plot 2: Macro-F1 curves
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_macro_f1", []), label="Train Macro-F1")
    plt.plot(epochs, metrics.get("val_macro_f1", []), label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Train vs Validation Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 curve: {e}")
    plt.close()

# ------------------------------- #
# Plot 3: Confusion matrix (last epoch)
try:
    preds_list = data.get("predictions", [])
    gts_list = data.get("ground_truth", [])
    if preds_list and gts_list:
        preds = preds_list[-1]
        gts = gts_list[-1]
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("SPR_BENCH: Confusion Matrix (Dev, Last Epoch)")
        plt.colorbar()
        classes = range(cm.shape[0])
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------- #
# Print best validation Macro-F1
try:
    val_f1s = metrics.get("val_macro_f1", [])
    if val_f1s:
        best_epoch = int(np.argmax(val_f1s)) + 1
        print(f"Best Validation Macro-F1: {max(val_f1s):.4f} at epoch {best_epoch}")
except Exception as e:
    print(f"Error computing best metric: {e}")

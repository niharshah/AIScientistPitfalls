import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested keys
def _get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


ed = _get(experiment_data, "Remove_Transformer_Encoder", "SPR_BENCH", default={})

if not ed:
    print("No data found for Remove_Transformer_Encoder / SPR_BENCH")
    exit()

epochs = ed.get("epochs", [])
loss_tr, loss_val = ed["losses"]["train"], ed["losses"]["val"]
f1_tr, f1_val = ed["metrics"]["train"], ed["metrics"]["val"]
test_preds, test_gts = ed.get("predictions", []), ed.get("ground_truth", [])

# ------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH – Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss Curves plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 2) Macro-F1 curves
try:
    plt.figure()
    plt.plot(epochs, f1_tr, label="Train")
    plt.plot(epochs, f1_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH – Macro-F1 Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 Curves plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 3) Confusion matrix on test set
try:
    if test_preds and test_gts:
        cm = confusion_matrix(test_gts, test_preds)
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(
            "SPR_BENCH – Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
        )
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating Confusion Matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------
# Print test metrics
print(f"Test loss  : {ed.get('test_loss', 'N/A')}")
print(f"Test MacroF1: {ed.get('test_macroF1', 'N/A')}")

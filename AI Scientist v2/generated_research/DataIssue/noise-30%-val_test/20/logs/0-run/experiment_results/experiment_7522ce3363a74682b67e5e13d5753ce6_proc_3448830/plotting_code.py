import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

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


# Helper to safely fetch nested keys
def get_path(d, *keys, default=None):
    for k in keys:
        if d is None or k not in d:
            return default
        d = d[k]
    return d


exp = get_path(experiment_data, "no_curriculum_weighting", "SPR_BENCH", default={})

loss_train = np.asarray(get_path(exp, "losses", "train", default=[]))
loss_val = np.asarray(get_path(exp, "losses", "val", default=[]))
val_metrics = get_path(exp, "metrics", "val", default=[])

macro_f1 = (
    np.asarray([m.get("macro_f1") for m in val_metrics])
    if val_metrics
    else np.array([])
)
cwa_arr = (
    np.asarray([m.get("cwa") for m in val_metrics]) if val_metrics else np.array([])
)

preds = np.asarray(get_path(exp, "predictions", default=[]))
labels = np.asarray(get_path(exp, "ground_truth", default=[]))

# ---------- plotting ----------
try:
    if loss_train.size and loss_val.size:
        plt.figure()
        epochs = np.arange(1, len(loss_train) + 1)
        plt.plot(epochs, loss_train, label="Train Loss")
        plt.plot(epochs, loss_val, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    if macro_f1.size:
        plt.figure()
        plt.plot(np.arange(1, len(macro_f1) + 1), macro_f1, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Validation Macro-F1 over Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 plot: {e}")
    plt.close()

try:
    if cwa_arr.size:
        plt.figure()
        plt.plot(np.arange(1, len(cwa_arr) + 1), cwa_arr, color="green", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.title("SPR_BENCH: Validation Complexity-Weighted Accuracy over Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

try:
    if preds.size and labels.size:
        cm = confusion_matrix(labels, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print best metrics ----------
if macro_f1.size:
    print(f"Best Macro-F1: {macro_f1.max():.3f}")
if cwa_arr.size:
    print(f"Best CWA:      {cwa_arr.max():.3f}")
print(f"Plots saved to {working_dir}")

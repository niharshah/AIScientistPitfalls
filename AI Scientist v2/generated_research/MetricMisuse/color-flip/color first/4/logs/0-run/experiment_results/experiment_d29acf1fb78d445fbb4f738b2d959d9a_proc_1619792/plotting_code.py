import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR", {})

metrics_tr = spr.get("metrics", {}).get("train", [])
metrics_val = spr.get("metrics", {}).get("val", [])
losses_tr = spr.get("losses", {}).get("train", [])
losses_val = spr.get("losses", {}).get("val", [])

epochs = range(1, len(losses_tr) + 1)


# Helper to extract metric arrays
def _metric_arr(metric_list, key):
    return [m.get(key, np.nan) for m in metric_list]


# ------------------- plotting --------------------
# 1. Loss curves
try:
    plt.figure()
    plt.plot(epochs, losses_tr, label="Train Loss")
    plt.plot(epochs, losses_val, label="Val Loss")
    plt.title("SPR: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. Accuracy curves
try:
    plt.figure()
    plt.plot(epochs, _metric_arr(metrics_tr, "accuracy"), label="Train Acc")
    plt.plot(epochs, _metric_arr(metrics_val, "accuracy"), label="Val Acc")
    plt.title("SPR: Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "SPR_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 3. PCWA curves
try:
    plt.figure()
    plt.plot(epochs, _metric_arr(metrics_tr, "pcwa"), label="Train PCWA")
    plt.plot(epochs, _metric_arr(metrics_val, "pcwa"), label="Val PCWA")
    plt.title("SPR: Training vs Validation PCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Pattern-Complexity Weighted Acc")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "SPR_PCWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PCWA curve: {e}")
    plt.close()

# 4. Confusion matrix heat-map for final epoch
try:
    import itertools
    from collections import Counter

    preds = spr.get("predictions", [])[-1]
    gts = spr.get("ground_truth", [])[-1]
    if preds and gts:
        num_classes = len(set(gts) | set(preds))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("SPR: Confusion Matrix (Val, Last Epoch)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix_epoch_last.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------- final metrics printout -------------------
if metrics_val:
    last = metrics_val[-1]
    print(
        f"Final Val Accuracy: {last.get('accuracy', 'n/a'):.3f}, "
        f"Final Val PCWA: {last.get('pcwa', 'n/a'):.3f}"
    )

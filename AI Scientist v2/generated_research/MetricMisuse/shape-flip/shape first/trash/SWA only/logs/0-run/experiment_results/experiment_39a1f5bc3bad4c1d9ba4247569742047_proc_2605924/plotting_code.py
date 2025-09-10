import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Navigate to FreezeEmb / SPR_BENCH if present
ed = {}
try:
    ed = experiment_data.get("FreezeEmb", {}).get("SPR_BENCH", {})
except Exception as e:
    print(f"Failed to extract experiment dictionary: {e}")


# Helper to safely fetch nested lists
def _get(path, default=None):
    cur = ed
    for p in path:
        cur = cur.get(p, {})
    return cur if cur else default


# Plot 1: Loss curves ------------------------------------------------ #
try:
    train_losses = ed.get("losses", {}).get("train", [])
    val_losses = ed.get("losses", {}).get("val", [])
    if train_losses and val_losses:
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating Loss plot: {e}")
    plt.close()

# Plot 2: Accuracy curves ------------------------------------------- #
try:
    train_acc = [m["acc"] for m in ed.get("metrics", {}).get("train", [])]
    val_acc = [m["acc"] for m in ed.get("metrics", {}).get("val", [])]
    if train_acc and val_acc:
        plt.figure()
        plt.plot(train_acc, label="Train Acc")
        plt.plot(val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating Accuracy plot: {e}")
    plt.close()

# Plot 3: Shape-weighted accuracy curves ---------------------------- #
try:
    train_swa = [m["swa"] for m in ed.get("metrics", {}).get("train", [])]
    val_swa = [m["swa"] for m in ed.get("metrics", {}).get("val", [])]
    if train_swa and val_swa:
        plt.figure()
        plt.plot(train_swa, label="Train SWA")
        plt.plot(val_swa, label="Val SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH Shape-Weighted Accuracy\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_swa_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# Plot 4: Confusion matrix on test set ------------------------------ #
try:
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])
    if preds and gts and len(preds) == len(gts):
        cm = np.zeros((2, 2), dtype=int)
        for p, t in zip(preds, gts):
            cm[t][p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            "SPR_BENCH Confusion Matrix\nLeft: Ground Truth rows, Right: Predictions cols"
        )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating Confusion Matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Print final evaluation metrics if available
test_metrics = ed.get("metrics", {}).get("test", {})
if test_metrics:
    print(
        f"Test Accuracy: {test_metrics.get('acc'):.3f} | "
        f"Test Shape-Weighted Accuracy: {test_metrics.get('swa'):.3f}"
    )

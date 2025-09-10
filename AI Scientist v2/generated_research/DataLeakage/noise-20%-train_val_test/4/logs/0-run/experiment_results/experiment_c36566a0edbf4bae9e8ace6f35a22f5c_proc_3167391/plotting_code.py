import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})
epochs = np.asarray(spr.get("epochs", []))


def arr(path, default=[]):
    cur = spr
    for p in path:
        cur = cur.get(p, {})
    return np.asarray(cur if isinstance(cur, list) else default)


# 1) Loss curves -------------------------------------------------------------
try:
    if epochs.size:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, arr(["losses", "train"]), label="train")
        plt.title("Left: Training Loss - SPR_BENCH")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, arr(["losses", "val"]), label="val", color="orange")
        plt.title("Right: Validation Loss - SPR_BENCH")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) F1 curves ---------------------------------------------------------------
try:
    if epochs.size:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, arr(["metrics", "train_f1"]), label="train")
        plt.title("Left: Training Macro-F1 - SPR_BENCH")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, arr(["metrics", "val_f1"]), label="val", color="orange")
        plt.title("Right: Validation Macro-F1 - SPR_BENCH")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# 3) Confusion matrix --------------------------------------------------------
try:
    preds = np.asarray(spr.get("predictions", []))
    gts = np.asarray(spr.get("ground_truth", []))
    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.title("Test Confusion Matrix - SPR_BENCH")
        plt.colorbar(im, fraction=0.046)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 4) Print final metric ------------------------------------------------------
test_f1 = spr.get("metrics", {}).get("test_f1", None)
if test_f1 is not None:
    print(f"Final Test Macro-F1: {test_f1:.4f}")

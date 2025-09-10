import matplotlib.pyplot as plt
import numpy as np
import os

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

ed = experiment_data.get("count_token_transformer", {})
epochs = ed.get("epochs", [])


# Helper fetcher
def arr(key1, key2):
    return np.asarray(ed.get(key1, {}).get(key2, []))


# 1) Loss curves
try:
    plt.figure(figsize=(10, 4))
    # Left: training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, arr("losses", "train"), label="Train")
    plt.title("Left: Training Loss - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Right: validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, arr("losses", "val"), label="Val", color="orange")
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

# 2) F1 curves
try:
    plt.figure(figsize=(10, 4))
    # Left: training F1
    plt.subplot(1, 2, 1)
    plt.plot(epochs, arr("metrics", "train_f1"), label="Train")
    plt.title("Left: Training Macro-F1 - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    # Right: validation F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, arr("metrics", "val_f1"), label="Val", color="orange")
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

# 3) Confusion matrix on test set
try:
    preds = np.asarray(ed.get("predictions", []))
    gts = np.asarray(ed.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = max(preds.max(), gts.max()) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            f"Confusion Matrix - SPR_BENCH\nTest Macro-F1 = {ed.get('metrics', {}).get('test_f1', np.nan):.4f}"
        )
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    else:
        print("No predictions/ground-truth available for confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# Print final test metric
test_f1 = ed.get("metrics", {}).get("test_f1", None)
if test_f1 is not None:
    print(f"Final Test Macro-F1: {test_f1:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# setup + load
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

cbc = experiment_data.get("char_bigram_count", {})

epochs = np.asarray(cbc.get("epochs", []))
train_loss = np.asarray(cbc.get("losses", {}).get("train", []))
val_loss = np.asarray(cbc.get("losses", {}).get("val", []))
train_f1 = np.asarray(cbc.get("metrics", {}).get("train_f1", []))
val_f1 = np.asarray(cbc.get("metrics", {}).get("val_f1", []))
test_f1 = cbc.get("metrics", {}).get("test_f1", None)
preds = np.asarray(cbc.get("predictions", []))
gts = np.asarray(cbc.get("ground_truth", []))

# ------------------------------------------------------------------
# 1) Loss curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Left: training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.title("Left: Training Loss - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Right: validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_loss, label="val", color="orange")
    plt.title("Right: Validation Loss - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_char_bigram_count_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) F1 curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Left: training F1
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_f1, label="train")
    plt.title("Left: Training Macro-F1 - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    # Right: validation F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1, label="val", color="orange")
    plt.title("Right: Validation Macro-F1 - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_char_bigram_count_f1_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion-matrix heat-map (at most 5×5 shown)
# ------------------------------------------------------------------
try:
    if preds.size and gts.size:
        n_cls = max(int(preds.max()), int(gts.max())) + 1
        mat = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            mat[int(t), int(p)] += 1
        view = mat[:5, :5]  # keep figure small if many classes
        plt.figure(figsize=(4, 4))
        im = plt.imshow(view, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Confusion Matrix (Top-5 classes) - SPR_BENCH")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_char_bigram_count_confusion.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------
# print evaluation metric
# ------------------------------------------------------------------
if test_f1 is not None:
    print(f"Final Test Macro-F1: {test_f1:.4f}")

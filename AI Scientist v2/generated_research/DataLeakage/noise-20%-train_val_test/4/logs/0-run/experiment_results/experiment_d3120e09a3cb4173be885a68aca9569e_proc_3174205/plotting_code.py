import matplotlib.pyplot as plt
import numpy as np
import os

# ensure working directory exists
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("bigram_only", {})
epochs = np.array(ed.get("epochs", []))
tr_loss = np.array(ed.get("losses", {}).get("train", []))
val_loss = np.array(ed.get("losses", {}).get("val", []))
tr_f1 = np.array(ed.get("metrics", {}).get("train_f1", []))
val_f1 = np.array(ed.get("metrics", {}).get("val_f1", []))
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# ------------------------------------------------------------------
# 1) Loss curves
# ------------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("bigram_only: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "bigram_only_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) F1 curves
# ------------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, tr_f1, label="Train Macro-F1")
    plt.plot(epochs, val_f1, label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("bigram_only: Training vs Validation Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "bigram_only_f1_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion matrix (only if predictions exist)
# ------------------------------------------------------------------
try:
    if preds.size and gts.size:
        num_labels = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("bigram_only: Test Confusion Matrix")
        plt.tight_layout()
        fname = os.path.join(working_dir, "bigram_only_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

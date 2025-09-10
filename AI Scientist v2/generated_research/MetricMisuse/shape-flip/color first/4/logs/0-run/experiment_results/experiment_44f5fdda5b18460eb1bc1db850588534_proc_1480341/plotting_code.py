import matplotlib.pyplot as plt
import numpy as np
import os

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

spr = experiment_data.get("SPR_BENCH", {})
losses = spr.get("losses", {})
metrics = spr.get("metrics", {})
preds = np.array(spr.get("predictions", []))
gts = np.array(spr.get("ground_truth", []))

# ----------------- Plot 1: Train vs Val Loss ----------------------
try:
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if train_loss and val_loss:
        epochs = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------------- Plot 2: Validation CWA over Epochs ---------------
try:
    val_cwa = metrics.get("val", [])
    if val_cwa:
        epochs = range(1, len(val_cwa) + 1)
        plt.figure()
        plt.plot(epochs, val_cwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR_BENCH Validation CWA Across Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_val_CWA.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# --------------- Plot 3: Confusion Matrix Heatmap ----------------
try:
    if preds.size and gts.size:
        num_classes = int(max(max(preds), max(gts)) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------------- Evaluation Metric -------------------------
try:
    if preds.size and gts.size:
        acc = (preds == gts).mean()
        print(f"Overall Classification Accuracy: {acc:.4f}")
except Exception as e:
    print(f"Error computing accuracy: {e}")

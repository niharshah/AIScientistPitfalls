import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------- setup & data loading ---------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Get relevant subtree if available
exp = experiment_data.get("UniLSTM", {}).get("SPR_BENCH", {})
losses = exp.get("losses", {})
metrics = exp.get("metrics", {})
preds = exp.get("predictions", [])
gts = exp.get("ground_truth", [])

# --------------------------- PLOT 1: losses --------------------------- #
try:
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if train_loss and val_loss:
        plt.figure()
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs. Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_losses.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------- PLOT 2: metrics ------------------------- #
try:
    val_metrics = metrics.get("val", [])
    if val_metrics:
        cwa = [m.get("CWA", 0) for m in val_metrics]
        swa = [m.get("SWA", 0) for m in val_metrics]
        gcwa = [m.get("GCWA", 0) for m in val_metrics]
        epochs = range(1, len(val_metrics) + 1)
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, gcwa, label="GCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Metrics Over Epochs")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# --------------------- PLOT 3: confusion matrix --------------------- #
try:
    if preds and gts:
        num_classes = len(set(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test)")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------------- print final test metrics ---------------------- #
test_metrics = metrics.get("test", {})
if test_metrics:
    print(
        "Test Metrics -> " + ", ".join(f"{k}: {v:.3f}" for k, v in test_metrics.items())
    )

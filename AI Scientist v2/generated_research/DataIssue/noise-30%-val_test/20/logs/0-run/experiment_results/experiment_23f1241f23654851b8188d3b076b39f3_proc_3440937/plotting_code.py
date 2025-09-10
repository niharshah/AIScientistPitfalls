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
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

losses = data.get("losses", {})
metrics = data.get("metrics", {})
preds = np.array(data.get("predictions", []))
gts = np.array(data.get("ground_truth", []))

epochs = np.arange(1, len(losses.get("train", [])) + 1)

# Plot 1: Loss curves
try:
    plt.figure()
    if len(losses.get("train", [])):
        plt.plot(epochs, losses["train"], label="Train Loss")
    if len(losses.get("val", [])):
        plt.plot(epochs, losses["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot 2: Validation Macro-F1
try:
    val_f1 = metrics.get("val", [])
    if len(val_f1):
        plt.figure()
        plt.plot(epochs, val_f1, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(
            "SPR_BENCH Validation Macro-F1\nLeft: Ground Truth, Right: Generated Samples"
        )
        fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating macro-F1 plot: {e}")
    plt.close()

# Plot 3: Confusion Matrix
try:
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=8,
                )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# Print final metric for quick reference
if len(metrics.get("val", [])):
    print(f"Final validation Macro-F1: {metrics['val'][-1]:.4f}")

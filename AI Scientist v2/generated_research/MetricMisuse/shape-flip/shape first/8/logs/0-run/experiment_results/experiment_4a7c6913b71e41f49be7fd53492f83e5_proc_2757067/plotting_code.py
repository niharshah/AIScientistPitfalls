import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["length_normalized_histogram"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = range(1, len(ed["losses"]["train"]) + 1)

# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
    plt.plot(epochs, ed["losses"]["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Shape-Weighted Accuracy curves
try:
    plt.figure()
    plt.plot(epochs, ed["metrics"]["train_swa"], label="Train SWA")
    plt.plot(epochs, ed["metrics"]["val_swa"], label="Validation SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH: Training vs Validation SWA")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_swa_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# 3) Ground-truth vs Prediction distribution on test set
try:
    gt = np.array(ed["ground_truth"])
    preds = np.array(ed["predictions"])
    counts_gt = [np.sum(gt == 0), np.sum(gt == 1)]
    counts_pred = [np.sum(preds == 0), np.sum(preds == 1)]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    axes[0].bar([0, 1], counts_gt, color="steelblue")
    axes[0].set_title("Ground Truth")
    axes[0].set_xticks([0, 1])
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    axes[1].bar([0, 1], counts_pred, color="orange")
    axes[1].set_title("Predictions")
    axes[1].set_xticks([0, 1])
    axes[1].set_xlabel("Class")

    fig.suptitle("SPR_BENCH Test Set – Left: Ground Truth, Right: Predictions")
    fname = os.path.join(working_dir, "spr_bench_gt_vs_pred.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating GT vs Pred plot: {e}")
    plt.close()

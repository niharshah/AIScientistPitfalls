import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------- helper to fetch the single run --------
if experiment_data:
    exp_key = next(iter(experiment_data))  # 'NoAugmentationContrastivePretrain'
    ds_key = next(iter(experiment_data[exp_key]))  # 'SPR_BENCH'
    run = experiment_data[exp_key][ds_key]
else:
    run = {}

# ------------- PLOT 1: loss curves -----------------
try:
    tr_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    epochs = range(1, len(tr_loss) + 1)
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------- PLOT 2: CompWA metric ---------------
try:
    compwa = run["metrics"]["val"]
    epochs = range(1, len(compwa) + 1)
    plt.figure()
    plt.plot(epochs, compwa, marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Comp-Weighted Accuracy")
    plt.title("SPR_BENCH Validation Comp-Weighted Accuracy")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_compWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CompWA curve: {e}")
    plt.close()

# ------------- PLOT 3: GT vs Pred distribution -----
try:
    gt = run["ground_truth"]
    preds = run["predictions"]
    if gt and preds:
        classes = sorted(set(gt) | set(preds))
        gt_counts = [gt.count(c) for c in classes]
        pr_counts = [preds.count(c) for c in classes]
        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pr_counts, width, label="Predictions")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title(
            "SPR_BENCH Class Distribution\nLeft: Ground Truth, Right: Generated Predictions"
        )
        plt.xticks(x, classes)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()

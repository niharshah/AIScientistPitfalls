import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
exp_key = "FreezeEncoder"
exp = experiment_data.get(exp_key, {}).get(ds_key, {})

loss_train = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
ccwa_val = exp.get("metrics", {}).get("val_CCWA", [])
preds = exp.get("predictions", [])
gts = exp.get("ground_truth", [])

# ------------------- Plot 1: Loss curves ---------------------
try:
    plt.figure()
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("FreezeEncoder on SPR_BENCH\nTraining vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------- Plot 2: CCWA curve ----------------------
try:
    plt.figure()
    plt.plot(epochs, ccwa_val, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("CCWA")
    plt.ylim(0, 1)
    plt.title("FreezeEncoder on SPR_BENCH\nValidation CCWA Across Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_CCWA_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

# ------------------- Plot 3: Class dist. ---------------------
try:
    if preds and gts:
        final_preds = np.array(preds[-1])
        final_gts = np.array(gts[-1])
        classes = sorted(set(final_gts) | set(final_preds))
        counts_gt = [np.sum(final_gts == c) for c in classes]
        counts_pr = [np.sum(final_preds == c) for c in classes]

        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, counts_gt, width, label="Ground Truth")
        plt.bar(x + width / 2, counts_pr, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(x, classes)
        plt.title(
            "FreezeEncoder on SPR_BENCH\nFinal-Epoch Class Distribution: GT vs Pred"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()

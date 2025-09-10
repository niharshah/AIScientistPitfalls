import matplotlib.pyplot as plt
import numpy as np
import os

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

plots_made = []


# Helper to extract arrays
def to_np(tups):
    if not tups:
        return np.array([]), np.array([])
    ep, val = zip(*tups)
    return np.array(ep), np.array(val)


exp = experiment_data.get("no_sym_branch", {}).get("SPR_BENCH", {})

# 1. Loss curves -------------------------------------------------------------
try:
    tr_epochs, tr_losses = to_np(exp.get("losses", {}).get("train", []))
    val_epochs, val_losses = to_np(exp.get("losses", {}).get("val", []))
    if tr_epochs.size and val_epochs.size:
        plt.figure()
        plt.plot(tr_epochs, tr_losses, label="Train")
        plt.plot(val_epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plots_made.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2. Shape-weighted accuracy curves -----------------------------------------
try:
    tr_epochs, tr_swa = to_np(exp.get("metrics", {}).get("train", []))
    val_epochs, val_swa = to_np(exp.get("metrics", {}).get("val", []))
    if tr_epochs.size and val_epochs.size:
        plt.figure()
        plt.plot(tr_epochs, tr_swa, label="Train")
        plt.plot(val_epochs, val_swa, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plots_made.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 3. Confusion matrix --------------------------------------------------------
try:
    preds = np.array(exp.get("predictions", []))
    gts = np.array(exp.get("ground_truth", []))
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plots_made.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# Report saved plots
for p in plots_made:
    print(f"Saved plot: {p}")

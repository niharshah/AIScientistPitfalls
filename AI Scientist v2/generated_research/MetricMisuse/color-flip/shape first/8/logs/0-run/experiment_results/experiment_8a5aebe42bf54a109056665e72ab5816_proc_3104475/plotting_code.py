import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Navigate keys safely
try:
    exp_key = next(iter(experiment_data.keys()))
    dset_key = next(iter(experiment_data[exp_key].keys()))
    ed = experiment_data[exp_key][dset_key]
except Exception as e:
    print(f"Error extracting experiment dict: {e}")
    ed = {}

epochs = list(range(1, len(ed.get("losses", {}).get("train", [])) + 1))

# Plot 1: train vs val loss
try:
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
    plt.plot(epochs, ed["losses"]["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        f"{dset_key} – Loss Curves\nMask-Only Augmentation (Structure-Preserving)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{dset_key}_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot 2: validation CCWA
try:
    plt.figure()
    plt.plot(epochs, ed["metrics"]["val_CCWA"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("CCWA")
    plt.title(f"{dset_key} – Validation CCWA over Epochs")
    plt.savefig(os.path.join(working_dir, f"{dset_key}_val_CCWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()


# Helper to build confusion matrix
def confusion_matrix(gt, pred, n):
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    return cm


# Determine class count
n_cls = 0
if ed.get("ground_truth"):
    n_cls = (
        max(
            max(max(g) for g in ed["ground_truth"]),
            max(max(p) for p in ed["predictions"]),
        )
        + 1
    )

# Plot 3 & 4: confusion matrices for first and last epoch (max 2 figs)
for idx, ep in enumerate([0, len(epochs) - 1][:2]):
    try:
        gt = ed["ground_truth"][ep]
        pr = ed["predictions"][ep]
        cm = confusion_matrix(gt, pr, n_cls)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{dset_key} – Confusion Matrix (Epoch {ep+1})")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        fname = f"{dset_key}_confusion_epoch_{ep+1}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for epoch {ep+1}: {e}")
        plt.close()

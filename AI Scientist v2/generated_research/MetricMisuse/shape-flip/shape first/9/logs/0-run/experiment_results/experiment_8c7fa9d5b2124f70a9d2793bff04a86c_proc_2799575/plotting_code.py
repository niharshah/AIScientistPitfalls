import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------  Load experiment data  ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    train_losses = spr["losses"]["train"]
    val_losses = spr["losses"]["val"]
    val_swa = spr["metrics"]["val"]
    test_swa = spr["metrics"]["test"]
    preds = np.array(spr["predictions"])
    gts = np.array(spr["ground_truth"])
    epochs = np.arange(1, len(train_losses) + 1)

    # 1. Loss curves --------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, linestyle="--", label="Validation")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. Validation SWA curves ---------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_swa, marker="o")
        plt.title("SPR_BENCH Validation Shape-Weighted-Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve plot: {e}")
        plt.close()

    # 3. Confusion matrix on test set --------------------------------------
    try:
        classes = sorted(set(gts))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix - Test Set")
        plt.xticks(classes)
        plt.yticks(classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in classes:
            for j in classes:
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
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # -----------------------  Print evaluation metric  --------------------
    print(f"Test Shape-Weighted-Accuracy: {test_swa:.4f}")
else:
    print("No data available to plot.")

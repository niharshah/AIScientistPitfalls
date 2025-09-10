import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["NoProj"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # ------------------ 1. Loss curves -----------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves (NoProj)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------ 2. Validation metric -----------------------
    try:
        val_metric = ed["metrics"]["val"]
        if any(v is not None for v in val_metric):
            plt.figure()
            plt.plot(epochs, val_metric, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title("SPR_BENCH Validation SWA (NoProj)")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_BENCH_val_metric_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating metric curve: {e}")
        plt.close()

    # ------------------ 3. Confusion matrix ------------------------
    try:
        gt = np.array(ed["ground_truth"])
        pr = np.array(ed["predictions"])
        if gt.size and pr.size:
            labels = np.unique(gt)
            n = len(labels)
            cm = np.zeros((n, n), dtype=int)
            for t, p in zip(gt, pr):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(labels, labels)
            plt.yticks(labels, labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_BENCH Confusion Matrix (NoProj)")
            for i in range(n):
                for j in range(n):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------------- print final evaluation metric -----------------
    try:
        print(f"Test Shape-Weighted Accuracy: {ed['metrics']['test']:.4f}")
    except Exception as e:
        print(f"Could not print test metric: {e}")

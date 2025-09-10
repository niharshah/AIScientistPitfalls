import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["NoWeightDecay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

# -------------------------------------------------------------------------
if exp is not None:
    # ------------------ 1. Loss curves -----------------------------------
    try:
        plt.figure()
        epochs = range(1, len(exp["losses"]["train"]) + 1)
        plt.plot(epochs, exp["losses"]["train"], label="Train")
        plt.plot(epochs, exp["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------ 2. MCC curves ------------------------------------
    try:
        plt.figure()
        epochs = range(1, len(exp["metrics"]["train"]) + 1)
        plt.plot(epochs, exp["metrics"]["train"], label="Train")
        plt.plot(epochs, exp["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews CorrCoef")
        plt.title("SPR_BENCH MCC Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve plot: {e}")
        plt.close()

    # ------------------ 3. Confusion-matrix style bars -------------------
    try:
        preds = exp["predictions"][-1].astype(int)
        gts = exp["ground_truth"][-1].astype(int)
        tp = np.sum((preds == 1) & (gts == 1))
        fp = np.sum((preds == 1) & (gts == 0))
        tn = np.sum((preds == 0) & (gts == 0))
        fn = np.sum((preds == 0) & (gts == 1))
        counts = [tp, fp, tn, fn]
        labels = ["TP", "FP", "TN", "FN"]
        plt.figure()
        plt.bar(labels, counts, color=["g", "r", "b", "orange"])
        plt.ylabel("Count")
        plt.title("SPR_BENCH Confusion Matrix Bars\nRight: Generated Predictions")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_bars.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion bar plot: {e}")
        plt.close()

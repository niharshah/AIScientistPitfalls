import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["no_class_weight"]["SPR_BENCH"]
    tr_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    tr_mcc = ed["metrics"]["train"]
    val_mcc = ed["metrics"]["val"]

    # ------------------------ plot 1: loss curves ----------------------------- #
    try:
        plt.figure()
        plt.plot(tr_loss, label="Train")
        plt.plot(val_loss, label="Validation")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch index")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------------ plot 2: MCC curves ------------------------------ #
    try:
        plt.figure()
        plt.plot(tr_mcc, label="Train")
        plt.plot(val_mcc, label="Validation")
        plt.title("SPR_BENCH: Training vs Validation MCC")
        plt.xlabel("Epoch index")
        plt.ylabel("Matthews Correlation Coefficient")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_mcc_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # ------------------- plot 3: confusion matrix bars ------------------------ #
    try:
        preds = ed["predictions"][-1].astype(int)  # most recent run
        gts = ed["ground_truth"][-1].astype(int)

        tp = int(((preds == 1) & (gts == 1)).sum())
        tn = int(((preds == 0) & (gts == 0)).sum())
        fp = int(((preds == 1) & (gts == 0)).sum())
        fn = int(((preds == 0) & (gts == 1)).sum())

        mcc_num = tp * tn - fp * fn
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den else 0.0
        print(f"Test MCC for last run: {mcc:.4f}")

        plt.figure()
        plt.bar(["TP", "FP", "FN", "TN"], [tp, fp, fn, tn], color=["g", "r", "r", "g"])
        plt.title("SPR_BENCH: Test Confusion-Matrix Counts")
        plt.ylabel("Count")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_counts.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion-matrix plot: {e}")
        plt.close()

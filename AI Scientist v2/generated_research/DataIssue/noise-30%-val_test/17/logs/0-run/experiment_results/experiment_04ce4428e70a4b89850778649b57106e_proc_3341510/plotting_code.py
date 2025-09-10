import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef, f1_score

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# proceed only if data loaded
if "SPR_BENCH" in experiment_data:
    ed = experiment_data["SPR_BENCH"]

    # ---------------- PLOT 1: loss curves ----------------
    try:
        plt.figure()
        epochs = range(1, len(ed["losses"]["train"]) + 1)
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH - Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------------- PLOT 2: MCC curves ----------------
    try:
        plt.figure()
        epochs = range(1, len(ed["metrics"]["train"]) + 1)
        plt.plot(epochs, ed["metrics"]["train"], label="Train MCC")
        plt.plot(epochs, ed["metrics"]["val"], label="Val MCC")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews CorrCoef")
        plt.title("SPR_BENCH - Training vs Validation MCC")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_MCC_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # identify best run (highest val-MCC stored in configs)
    best_idx = int(np.argmax([c["best_val_mcc"] for c in ed["configs"]]))
    preds = ed["predictions"][best_idx].astype(int)
    gts = ed["ground_truth"][best_idx].astype(int)

    # ---------------- PLOT 3: confusion bar ----------------
    try:
        tp = int(np.sum((preds == 1) & (gts == 1)))
        fp = int(np.sum((preds == 1) & (gts == 0)))
        tn = int(np.sum((preds == 0) & (gts == 0)))
        fn = int(np.sum((preds == 0) & (gts == 1)))
        plt.figure()
        plt.bar(
            ["TP", "FP", "TN", "FN"], [tp, fp, tn, fn], color=["g", "r", "b", "orange"]
        )
        plt.title("SPR_BENCH - Confusion Counts (Best Run)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_counts.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion plot: {e}")
        plt.close()

    # ----------- print evaluation metrics -------------
    best_mcc = matthews_corrcoef(gts, preds)
    best_f1 = f1_score(gts, preds, average="macro")
    print(f"Best run metrics -> Test MCC: {best_mcc:.4f}, Test Macro-F1: {best_f1:.4f}")
else:
    print("SPR_BENCH data not found in experiment_data.")

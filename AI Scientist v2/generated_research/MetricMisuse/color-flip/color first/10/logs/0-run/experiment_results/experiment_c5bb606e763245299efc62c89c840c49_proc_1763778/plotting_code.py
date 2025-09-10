import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- basic setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    try:
        md = experiment_data["Factorized_SC"]["SPR_BENCH"]["metrics"]
        preds = np.array(experiment_data["Factorized_SC"]["SPR_BENCH"]["predictions"])
        gts = np.array(experiment_data["Factorized_SC"]["SPR_BENCH"]["ground_truth"])
        epochs = np.arange(1, len(md["train_loss"]) + 1)
    except KeyError as e:
        print(f"Key error parsing experiment data: {e}")
        md, preds, gts, epochs = None, None, None, None

# ----------------- plot 1: loss curves ----------------
if md is not None:
    try:
        plt.figure()
        plt.plot(epochs, md["train_loss"], label="Train loss")
        plt.plot(epochs, md["val_loss"], label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

# ----------- plot 2: validation accuracies ------------
if md is not None:
    try:
        plt.figure()
        plt.plot(epochs, md["val_CWA"], label="CWA")
        plt.plot(epochs, md["val_SWA"], label="SWA")
        plt.plot(epochs, md["val_CWA2"], label="CWA2")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Validation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_weighted_acc.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

# ----------- plot 3: confusion matrix bars ------------
if preds is not None and gts is not None:
    try:
        tp = np.sum((gts == 1) & (preds == 1))
        tn = np.sum((gts == 0) & (preds == 0))
        fp = np.sum((gts == 0) & (preds == 1))
        fn = np.sum((gts == 1) & (preds == 0))
        counts = [tp, tn, fp, fn]
        labels = ["TP", "TN", "FP", "FN"]
        plt.figure()
        plt.bar(labels, counts, color=["g", "b", "r", "orange"])
        plt.title("SPR_BENCH Confusion Matrix Counts")
        for i, v in enumerate(counts):
            plt.text(i, v + max(counts) * 0.01, str(v), ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_counts.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion bar chart: {e}")
        plt.close()

# --------------- print final metrics ------------------
if md is not None:
    print(f"Final validation loss  : {md['val_loss'][-1]:.4f}")
    print(f"Final CWA             : {md['val_CWA'][-1]:.4f}")
    print(f"Final SWA             : {md['val_SWA'][-1]:.4f}")
    print(f"Final CWA2            : {md['val_CWA2'][-1]:.4f}")

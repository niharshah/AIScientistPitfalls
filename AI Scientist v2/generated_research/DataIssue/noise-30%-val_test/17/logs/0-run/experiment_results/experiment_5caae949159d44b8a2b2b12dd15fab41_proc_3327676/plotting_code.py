import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data.get("SPR_BENCH", {})
    train_losses = data.get("losses", {}).get("train", [])
    val_losses = data.get("losses", {}).get("val", [])
    train_mcc = data.get("metrics", {}).get("train", [])
    val_mcc = data.get("metrics", {}).get("val", [])
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- Plot 2: MCC curves ----------
    try:
        plt.figure()
        epochs = np.arange(1, len(train_mcc) + 1)
        plt.plot(epochs, train_mcc, label="Train MCC")
        plt.plot(epochs, val_mcc, label="Val MCC")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("SPR_BENCH Training vs Validation MCC")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_mcc_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # ---------- Plot 3: Confusion-matrix style bar plot ----------
    try:
        plt.figure()
        tp = np.sum((preds == 1) & (gts == 1))
        tn = np.sum((preds == 0) & (gts == 0))
        fp = np.sum((preds == 1) & (gts == 0))
        fn = np.sum((preds == 0) & (gts == 1))
        bars = [tp, fp, fn, tn]
        labels = ["TP", "FP", "FN", "TN"]
        plt.bar(labels, bars, color=["g", "r", "r", "g"])
        plt.ylabel("Count")
        plt.title("SPR_BENCH Test Confusion Matrix (bar)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion plot: {e}")
        plt.close()

    # ---------- Print key metrics ----------
    if val_mcc:
        print(f"Final Validation MCC: {val_mcc[-1]:.4f}")
    if preds.size and gts.size:
        test_mcc = matthews_corrcoef(gts, preds)
        print(f"Test MCC: {test_mcc:.4f}")

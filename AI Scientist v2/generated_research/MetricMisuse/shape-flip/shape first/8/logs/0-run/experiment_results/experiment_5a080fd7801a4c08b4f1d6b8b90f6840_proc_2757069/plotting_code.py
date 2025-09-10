import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = exp["linear_only"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # 1) Loss curves -----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR-BENCH Training vs Validation Loss\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2) SWA curves ------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_swa"], label="Train SWA")
        plt.plot(epochs, ed["metrics"]["val_swa"], label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR-BENCH Shape-Weighted Accuracy\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve plot: {e}")
        plt.close()

    # 3) Confusion matrix style bar plot ---------------------------------------
    try:
        y_true = np.array(ed["ground_truth"])
        y_pred = np.array(ed["predictions"])
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        counts = [tp, fp, fn, tn]
        labels = ["TP", "FP", "FN", "TN"]
        plt.figure()
        plt.bar(labels, counts, color=["green", "red", "red", "green"])
        plt.ylabel("Count")
        plt.title("SPR-BENCH Test Predictions vs Ground Truth\nConfusion-Matrix Counts")
        fname = os.path.join(working_dir, "spr_bench_confusion_counts.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

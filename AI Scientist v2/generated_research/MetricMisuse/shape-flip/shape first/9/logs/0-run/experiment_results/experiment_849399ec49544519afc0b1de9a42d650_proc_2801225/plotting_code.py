import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["LateFusion_LogitsAvg"]["SPR_BENCH"]
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    val_swa = ed["metrics"]["val"]
    preds = np.array(ed["predictions"])
    gts = np.array(ed["ground_truth"])
    epochs = np.arange(1, len(train_loss) + 1)

    # --------- Plot: Loss curves -------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.title("SPR_BENCH: Training vs. Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------- Plot: Validation SWA ----------
    try:
        plt.figure()
        plt.plot(epochs, val_swa, marker="o")
        plt.title("SPR_BENCH: Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.ylim(0, 1)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val SWA plot: {e}")
        plt.close()

    # --------- Plot: Confusion Matrix --------
    try:
        classes = np.unique(np.concatenate((gts, preds)))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(classes)
        plt.yticks(classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------- Print final metric ------------
    try:
        test_swa = ed["metrics"]["test"]
        print(f"Final Test Shape-Weighted Accuracy: {test_swa:.4f}")
    except Exception as e:
        print(f"Could not retrieve test metric: {e}")

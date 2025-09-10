import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------- load data -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    epochs = exp["epochs"]
    train_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    train_f1 = exp["metrics"]["train_f1"]
    val_f1 = exp["metrics"]["val_f1"]
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    test_f1 = exp["metrics"]["test_f1"]
    sga = exp["metrics"]["SGA"]

    # --------------------- loss curves ------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------------------- F1 curves -------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH F1 Curves\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # ------------------- confusion matrix --------------------
    try:
        if preds.size and gts.size:
            cm = confusion_matrix(gts, preds)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("SPR_BENCH Confusion Matrix\nTest Set")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            # annotate cells
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
                )
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------------------- print metrics --------------------
    print(f"Test Macro-F1: {test_f1:.4f} | SGA: {sga:.4f}")

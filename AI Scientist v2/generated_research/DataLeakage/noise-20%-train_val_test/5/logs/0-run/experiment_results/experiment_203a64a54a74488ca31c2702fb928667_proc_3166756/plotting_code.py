import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    tr_loss = spr["metrics"]["train"]["loss"]
    val_loss = spr["metrics"]["val"]["loss"]
    tr_f1 = spr["metrics"]["train"]["f1"]
    val_f1 = spr["metrics"]["val"]["f1"]
    test_loss = spr["metrics"]["test"].get("loss", None)
    test_f1 = spr["metrics"]["test"].get("f1", None)
    preds = np.array(spr["predictions"])
    gts = np.array(spr["ground_truth"])
    epochs = np.arange(1, len(tr_loss) + 1)

    # ------------------ 1. Loss curves ----------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train", marker="o")
        plt.plot(epochs, val_loss, label="Validation", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ------------------ 2. F1 curves ------------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train", marker="o")
        plt.plot(epochs, val_f1, label="Validation", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # ------------------ 3. Confusion matrix ------------------------ #
    try:
        num_classes = int(max(gts.max(), preds.max()) + 1) if gts.size else 0
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Bottom: Predicted")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------ 4. Print test metrics ---------------------- #
    print(f"Test Loss: {test_loss:.4f}  |  Test Macro-F1: {test_f1:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
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
    # Extract arrays
    tr_acc = np.array([d["acc"] for d in spr["metrics"]["train"]])
    val_acc = np.array([d["acc"] for d in spr["metrics"]["val"]])
    tr_loss = np.array(spr["losses"]["train"])
    val_loss = np.array(spr["losses"]["val"])
    epochs = np.arange(1, len(tr_acc) + 1)
    preds = np.array(spr["predictions"])
    gts = np.array(spr["ground_truth"])
    num_cls = len(np.unique(gts))

    # ---------------- Accuracy curves ----------------
    try:
        plt.figure()
        plt.plot(epochs, tr_acc, marker="o", label="Train")
        plt.plot(epochs, val_acc, marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Training vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

    # ---------------- Loss curves --------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, marker="o", label="Train")
        plt.plot(epochs, val_loss, marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------------- Confusion matrix ---------------
    try:
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------------- Metrics printout ---------------
    try:
        test_acc = (preds == gts).mean()
        # macro-F1
        f1s = []
        for c in range(num_cls):
            tp = ((preds == c) & (gts == c)).sum()
            fp = ((preds == c) & (gts != c)).sum()
            fn = ((preds != c) & (gts == c)).sum()
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
        macro_f1 = float(np.mean(f1s))
        print(f"Test Accuracy: {test_acc:.4f} | Macro-F1: {macro_f1:.4f}")
    except Exception as e:
        print(f"Error computing metrics: {e}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# Set up working directory & load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:
    epochs = np.arange(1, len(spr["metrics"]["train_acc"]) + 1)

    # ---------- 1. Accuracy curves ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["metrics"]["train_acc"], marker="o", label="Train")
        plt.plot(epochs, spr["metrics"]["val_acc"], marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- 2. Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], marker="o", label="Train")
        plt.plot(epochs, spr["losses"]["val"], marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 3. Macro-F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["metrics"]["train_f1"], marker="o", label="Train")
        plt.plot(epochs, spr["metrics"]["val_f1"], marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # ---------- 4. Confusion matrix ----------
    try:
        preds = np.array(spr["predictions"])
        gts = np.array(spr["ground_truth"])
        num_classes = len(np.unique(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix\nDataset: Test")
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

    # ---------- 5. Class distribution bar chart ----------
    try:
        plt.figure()
        width = 0.35
        classes = np.arange(num_classes)
        gt_counts = [np.sum(gts == c) for c in classes]
        pred_counts = [np.sum(preds == c) for c in classes]
        plt.bar(classes - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(classes + width / 2, pred_counts, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("SPR_BENCH Class Distribution\nLeft: GT, Right: Pred")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating class distribution plot: {e}")
        plt.close()

    # ---------- Print final metrics ----------
    try:
        test_acc = (preds == gts).mean()
        # Recompute macro-F1 quickly
        f1s = []
        for c in range(num_classes):
            tp = np.sum((preds == c) & (gts == c))
            fp = np.sum((preds == c) & (gts != c))
            fn = np.sum((preds != c) & (gts == c))
            if tp == 0 and (fp == 0 or fn == 0):
                f1 = 0.0
            else:
                prec = tp / (tp + fp + 1e-9)
                rec = tp / (tp + fn + 1e-9)
                f1 = 2 * prec * rec / (prec + rec + 1e-9)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Final Macro-F1:      {macro_f1:.4f}")
    except Exception as e:
        print(f"Error computing final metrics: {e}")

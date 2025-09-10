import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
exp_path = os.path.join(working_dir, "experiment_data.npy")

# -------------------------------------------------
try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    spr_data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

# -------------------------------------------------
if spr_data is not None:
    preds = spr_data["predictions"]
    gts = spr_data["ground_truth"]
    test_acc = (preds == gts).mean() if len(preds) else float("nan")
    print(f"SPR_BENCH test accuracy: {test_acc:.4f}")

# ---------- Accuracy curve ----------
try:
    if spr_data is None:
        raise ValueError("No data to plot")
    epochs = spr_data["epochs"]
    tr_acc = spr_data["metrics"]["train_acc"]
    va_acc = spr_data["metrics"]["val_acc"]

    plt.figure()
    plt.plot(epochs, tr_acc, label="Train")
    plt.plot(epochs, va_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curve\nTrain vs Validation")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- Loss curve ----------
try:
    if spr_data is None:
        raise ValueError("No data to plot")
    epochs = spr_data["epochs"]
    tr_loss = spr_data["losses"]["train"]
    va_loss = spr_data["losses"]["val"]

    plt.figure()
    plt.plot(epochs, tr_loss, label="Train")
    plt.plot(epochs, va_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curve\nTrain vs Validation")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- Confusion matrix ----------
try:
    if spr_data is None or len(preds) == 0:
        raise ValueError("No prediction data for confusion matrix")
    num_labels = int(max(preds.max(), gts.max()) + 1)
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("SPR_BENCH Confusion Matrix\nRows: Ground Truth, Cols: Predictions")
    plt.xticks(range(num_labels))
    plt.yticks(range(num_labels))
    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    save_path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

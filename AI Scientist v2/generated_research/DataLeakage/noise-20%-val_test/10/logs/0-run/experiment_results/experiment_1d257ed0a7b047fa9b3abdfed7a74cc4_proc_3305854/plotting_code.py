import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None


# ---------- helper ----------
def macro_f1(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lb in labels:
        tp = np.sum((y_true == lb) & (y_pred == lb))
        fp = np.sum((y_true != lb) & (y_pred == lb))
        fn = np.sum((y_true == lb) & (y_pred != lb))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return np.mean(f1s)


if spr:
    preds = np.array(spr["predictions"])
    gts = np.array(spr["ground_truth"])
    test_acc = (preds == gts).mean()
    m_f1 = macro_f1(gts, preds)
    print(
        f"SPR_BENCH  |  Test Accuracy: {test_acc:.4f}  |  Macro-F1: {m_f1:.4f}  |  REA: {spr['metrics']['REA']:.4f}"
    )

    # ---------- 1) Accuracy curves ----------
    try:
        epochs = np.arange(1, len(spr["metrics"]["train_acc"]) + 1)
        plt.figure()
        plt.plot(epochs, spr["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, spr["metrics"]["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_acc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- 2) Loss curves ----------
    try:
        epochs = np.arange(1, len(spr["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train Loss")
        plt.plot(epochs, spr["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 3) Confusion Matrix ----------
    try:
        labels = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.xticks(labels)
        plt.yticks(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

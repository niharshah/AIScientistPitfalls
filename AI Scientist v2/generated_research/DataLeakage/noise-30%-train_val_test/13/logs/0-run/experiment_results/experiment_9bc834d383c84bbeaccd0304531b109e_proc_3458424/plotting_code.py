import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None


def macro_f1_score(y_true, y_pred):
    labels = np.unique(y_true)
    f1s = []
    for l in labels:
        tp = np.sum((y_pred == l) & (y_true == l))
        fp = np.sum((y_pred == l) & (y_true != l))
        fn = np.sum((y_pred != l) & (y_true == l))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


if spr is not None:
    epochs = np.array(spr["epochs"])
    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train")
        plt.plot(epochs, spr["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- Plot 2: F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, spr["metrics"]["train_f1"], label="Train")
        plt.plot(epochs, spr["metrics"]["val_f1"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    preds = np.array(spr["predictions"])
    labels = np.array(spr["ground_truth"])

    # ---------- Plot 3: Confusion matrix ----------
    try:
        n_classes = int(max(labels.max(), preds.max()) + 1)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- Plot 4: Label distribution ----------
    try:
        gt_counts = np.bincount(labels, minlength=int(preds.max() + 1))
        pred_counts = np.bincount(preds, minlength=int(preds.max() + 1))
        idx = np.arange(len(gt_counts))
        width = 0.35
        plt.figure()
        plt.bar(idx - width / 2, gt_counts, width=width, label="Ground Truth")
        plt.bar(idx + width / 2, pred_counts, width=width, label="Predictions")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title("SPR_BENCH Label Distribution: Ground Truth vs Predictions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_label_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()

    # ---------- Print evaluation metric ----------
    print("Final Test Macro-F1:", macro_f1_score(labels, preds))

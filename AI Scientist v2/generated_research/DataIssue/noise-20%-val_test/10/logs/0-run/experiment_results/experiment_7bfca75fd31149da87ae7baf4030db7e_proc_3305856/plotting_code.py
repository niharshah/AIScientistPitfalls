import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
exp_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    train_f1 = np.array(data["metrics"]["train_f1"])
    val_f1 = np.array(data["metrics"]["val_f1"])
    train_loss = np.array(data["losses"]["train"])
    val_loss = np.array(data["losses"]["val"])
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    epochs = np.arange(1, len(train_f1) + 1)

    # helper for macro-F1
    def macro_f1(y_true, y_pred):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1_vals = []
        for lb in labels:
            tp = np.sum((y_true == lb) & (y_pred == lb))
            fp = np.sum((y_true != lb) & (y_pred == lb))
            fn = np.sum((y_true == lb) & (y_pred != lb))
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1_vals.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        return np.mean(f1_vals)

    print("Test Macro-F1:", macro_f1(gts, preds))

    # -------- 1) F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train F1")
        plt.plot(epochs, val_f1, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Train vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # -------- 2) Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- 3) Confusion matrix ----------
    try:
        labels = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Left: True, Top: Pred)")
        plt.xticks(labels)
        plt.yticks(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=7
                )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

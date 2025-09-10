import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    log = exp["no_interaction_symbolic"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    log = None

saved = []
if log:
    epochs = log["epochs"]
    tr_loss = log["losses"]["train"]
    va_loss = log["losses"]["val"]
    val_swa = log["metrics"]["val"]
    preds = log["predictions"]
    gts = log["ground_truth"]
    # best epoch idx
    best_idx = int(np.argmin(va_loss))

    # 1) loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, va_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH — Training vs Validation Loss")
        plt.legend()
        f = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(f)
        saved.append(f)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) validation accuracy curve
    try:
        plt.figure()
        plt.plot(epochs, val_swa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH — Validation SWA Across Epochs")
        f = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(f)
        saved.append(f)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 3) confusion matrix at best epoch
    try:
        y_true = np.array(gts[best_idx])
        y_pred = np.array(preds[best_idx])
        n_cls = len(set(y_true) | set(y_pred))
        cm = np.zeros((n_cls, n_cls), int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title(f"SPR_BENCH — Confusion Matrix (Best Epoch {epochs[best_idx]})")
        f = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(f)
        saved.append(f)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

print("Saved figures:", saved)

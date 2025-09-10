import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data -------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("MaxPoolSeqRep", {}).get("SPR_BENCH", {})


# helper to create epoch index
def epoch_idx(arr):
    return np.arange(1, len(arr) + 1)


# 1) train / val loss --------------------------------------------
try:
    train_loss = exp["metrics"]["train_loss"]
    val_loss = exp["metrics"]["val_loss"]
    if train_loss and val_loss:
        plt.figure()
        epochs = epoch_idx(train_loss)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) validation SWA ----------------------------------------------
try:
    val_swa = exp["metrics"]["val_swa"]
    if val_swa:
        plt.figure()
        epochs = epoch_idx(val_swa)
        plt.plot(epochs, val_swa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation SWA Over Epochs")
        save_path = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# 3) confusion matrix (test) -------------------------------------
try:
    preds = exp["predictions"]["test"]
    gtruth = exp["ground_truth"]["test"]
    if preds and gtruth:
        preds = np.array(preds, dtype=int)
        gtruth = np.array(gtruth, dtype=int)
        n_cls = int(max(preds.max(), gtruth.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gtruth, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("SPR_BENCH: Confusion Matrix (Test)")
        # annotate cells
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        save_path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_test.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

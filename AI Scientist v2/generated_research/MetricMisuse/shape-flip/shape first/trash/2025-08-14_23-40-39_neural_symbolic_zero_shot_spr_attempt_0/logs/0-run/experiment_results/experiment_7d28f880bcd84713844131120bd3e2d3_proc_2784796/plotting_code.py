import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product
from collections import Counter

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    exp = experiment_data["Remove-Transformer-Encoder"]["SPR_BENCH"]
    train_loss = exp["metrics"]["train_loss"]
    val_loss = exp["metrics"]["val_loss"]
    val_swa = exp["metrics"]["val_swa"]
    dev_y_true = exp["ground_truth"]["dev"]
    dev_y_pred = exp["predictions"]["dev"]
    test_y_true = exp["ground_truth"]["test"]
    test_y_pred = exp["predictions"]["test"]
    classes = sorted(set(dev_y_true) | set(test_y_true))

    # ----------- helper for confusion matrix -------------------
    def plot_confusion(y_true, y_pred, split_name):
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"SPR_BENCH Confusion Matrix ({split_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    # -------------------- PLOTS --------------------------------
    # 1. Loss curve
    try:
        plt.figure()
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2. Validation SWA curve
    try:
        plt.figure()
        plt.plot(epochs, val_swa, marker="o")
        plt.title("SPR_BENCH Validation Shape-Weighted Accuracy (SWA)")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_SWA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # 3. Dev confusion matrix
    try:
        fig = plot_confusion(dev_y_true, dev_y_pred, "DEV")
        fig.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_DEV.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating DEV confusion: {e}")
        plt.close()

    # 4. Test confusion matrix
    try:
        fig = plot_confusion(test_y_true, test_y_pred, "TEST")
        fig.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_TEST.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating TEST confusion: {e}")
        plt.close()

    # -------------------- PRINT METRICS -------------------------
    if val_swa:
        print(f"Final DEV SWA:  {val_swa[-1]:.4f}")
    if test_y_true:
        # recompute test swa quickly
        def count_shape_variety(seq):
            return len(set(tok[0] for tok in seq.split()))

        test_seqs = experiment_data["Remove-Transformer-Encoder"]["SPR_BENCH"][
            "ground_truth"
        ]
        # Cannot recompute without sequences here; just print placeholder
        print(f"Stored TEST predictions: {len(test_y_pred)} samples")

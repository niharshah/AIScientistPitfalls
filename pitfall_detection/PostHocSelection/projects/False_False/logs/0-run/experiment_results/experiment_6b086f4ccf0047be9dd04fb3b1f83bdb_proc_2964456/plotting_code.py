import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- data loading ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["weight_decay_tuning"]["SPR_BENCH"]
    wds = ed["configs"]
    tr_losses = ed["losses"]["train"]
    val_losses = ed["losses"]["val"]
    tr_f1 = ed["metrics"]["train"]
    val_f1 = ed["metrics"]["val"]

    # Identify best config by final val macro-F1
    final_val_f1 = [vals[-1] for vals in val_f1]
    best_idx = int(np.argmax(final_val_f1))
    best_wd = wds[best_idx]
    print(
        f"Best weight_decay={best_wd} with final Val Macro-F1={final_val_f1[best_idx]:.4f}"
    )

    # ---------------- plots ----------------
    # 1) Loss curves
    try:
        plt.figure()
        epochs = np.arange(1, len(tr_losses[0]) + 1)
        for i, wd in enumerate(wds):
            plt.plot(epochs, tr_losses[i], "--", label=f"train wd={wd}")
            plt.plot(epochs, val_losses[i], "-", label=f"val wd={wd}")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend(fontsize=8)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) Macro-F1 curves
    try:
        plt.figure()
        for i, wd in enumerate(wds):
            plt.plot(epochs, tr_f1[i], "--", label=f"train wd={wd}")
            plt.plot(epochs, val_f1[i], "-", label=f"val wd={wd}")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend(fontsize=8)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # 3) Scatter of final Val F1 vs weight decay
    try:
        plt.figure()
        plt.scatter(wds, final_val_f1, c="red")
        for wd, f1 in zip(wds, final_val_f1):
            plt.text(wd, f1, f"{f1:.2f}", fontsize=8, ha="center", va="bottom")
        plt.xscale("log")
        plt.xlabel("Weight Decay (log scale)")
        plt.ylabel("Final Validation Macro-F1")
        plt.title("SPR_BENCH: Final Val Macro-F1 vs Weight Decay")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_wd_vs_f1.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        plt.close()

    # 4) Confusion matrix for best model
    try:
        gt = ed["ground_truth"][best_idx]
        pred = ed["predictions"][best_idx]
        cm = confusion_matrix(gt, pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"SPR_BENCH Confusion Matrix (wd={best_wd})")
        plt.colorbar()
        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    lr_runs = experiment_data["learning_rate"]["SPR_BENCH"]
    lrs = sorted(lr_runs.keys(), key=float)
    epochs = len(next(iter(lr_runs.values()))["losses"]["train"])  # assume same length

    # utility to get colors
    cmap = plt.cm.get_cmap("tab10", len(lrs))

    # 1) Training loss curves
    try:
        plt.figure()
        for idx, lr in enumerate(lrs):
            plt.plot(
                range(1, epochs + 1),
                lr_runs[lr]["losses"]["train"],
                label=f"LR {lr}",
                color=cmap(idx),
            )
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("SPR_BENCH: Training Loss vs. Epoch")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_train_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating train loss plot: {e}")
        plt.close()

    # 2) Validation loss curves
    try:
        plt.figure()
        for idx, lr in enumerate(lrs):
            plt.plot(
                range(1, epochs + 1),
                lr_runs[lr]["losses"]["val"],
                label=f"LR {lr}",
                color=cmap(idx),
            )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("SPR_BENCH: Validation Loss vs. Epoch")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val loss plot: {e}")
        plt.close()

    # 3) Validation Macro-F1 curves
    try:
        plt.figure()
        for idx, lr in enumerate(lrs):
            plt.plot(
                range(1, epochs + 1),
                lr_runs[lr]["metrics"]["val"],
                label=f"LR {lr}",
                color=cmap(idx),
            )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Validation Macro-F1 vs. Epoch")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # 4) Final Macro-F1 bar chart
    try:
        final_f1 = [lr_runs[lr]["metrics"]["val"][-1] for lr in lrs]
        plt.figure()
        plt.bar(range(len(lrs)), final_f1, tick_label=lrs)
        plt.ylabel("Final Macro-F1")
        plt.title("SPR_BENCH: Final Macro-F1 per Learning Rate")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_final_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final F1 bar plot: {e}")
        plt.close()

    # 5) Confusion matrix for best LR
    try:
        best_idx = int(np.argmax(final_f1))
        best_lr = lrs[best_idx]
        gt = lr_runs[best_lr]["ground_truth"]
        preds = lr_runs[best_lr]["predictions"]
        cm = confusion_matrix(gt, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"SPR_BENCH: Confusion Matrix (Best LR {best_lr})")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_confusion_matrix_lr_{best_lr}.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # print evaluation metric
    print(f"Best LR: {best_lr} | Final Macro-F1: {final_f1[best_idx]:.4f}")

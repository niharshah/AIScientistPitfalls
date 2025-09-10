import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------#
# load experiment data                                                     #
# -------------------------------------------------------------------------#
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    runs = experiment_data["learning_rate"]["SPR_BENCH"]["runs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = []

# helper to ensure we have something to plot
if not runs:
    print("No runs found – exiting plotting script.")
else:
    # ---------------------------------------------------------------------#
    # 1. accuracy curves                                                   #
    # ---------------------------------------------------------------------#
    try:
        plt.figure()
        for run in runs:
            epochs = np.arange(1, len(run["metrics"]["train"]) + 1)
            plt.plot(
                epochs,
                run["metrics"]["train"],
                label=f"train lr={run['lr']:.0e}",
            )
            plt.plot(
                epochs,
                run["metrics"]["val"],
                linestyle="--",
                label=f"val lr={run['lr']:.0e}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy Curves")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

    # ---------------------------------------------------------------------#
    # 2. loss curves                                                       #
    # ---------------------------------------------------------------------#
    try:
        plt.figure()
        for run in runs:
            epochs = np.arange(1, len(run["losses"]["train"]) + 1)
            plt.plot(
                epochs,
                run["losses"]["train"],
                label=f"train lr={run['lr']:.0e}",
            )
            plt.plot(
                epochs,
                run["losses"]["val"],
                linestyle="--",
                label=f"val lr={run['lr']:.0e}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss Curves")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------------------------------------------------------------------#
    # 3. test accuracy bar plot                                            #
    # ---------------------------------------------------------------------#
    try:
        lrs = [run["lr"] for run in runs]
        test_accs = [run["test_acc"] for run in runs]
        plt.figure()
        plt.bar([f"{lr:.0e}" for lr in lrs], test_accs, color="skyblue")
        plt.xlabel("Learning Rate")
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH: Test Accuracy vs Learning Rate")
        fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar plot: {e}")
        plt.close()

    # ---------------------------------------------------------------------#
    # 4. confusion matrix for best LR                                      #
    # ---------------------------------------------------------------------#
    try:
        # pick run with highest test accuracy
        best_run = max(runs, key=lambda r: r["test_acc"])
        preds = np.asarray(best_run["predictions"])
        gts = np.asarray(best_run["ground_truth"])
        classes = np.unique(np.concatenate([preds, gts]))
        num_classes = len(classes)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(
            f"SPR_BENCH: Confusion Matrix (Best LR={best_run['lr']:.0e})\nLeft: Ground Truth, Right: Predictions"
        )
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        # annotate cells
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=6,
                )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

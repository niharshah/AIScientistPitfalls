import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------- #
# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# proceed only if data are present
if "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]

    # helper to safely fetch lists -> numpy arrays
    def _get(arr):
        try:
            return np.asarray(arr, dtype=float)
        except Exception:
            return np.array([])

    train_loss = _get(data["losses"].get("train", []))
    val_loss = _get(data["losses"].get("val", []))
    train_acc = _get(data["metrics"].get("train_acc", []))
    val_acc = _get(data["metrics"].get("val_acc", []))
    train_f1 = _get(data["metrics"].get("train_f1", []))
    val_f1 = _get(data["metrics"].get("val_f1", []))
    epochs = np.arange(1, len(train_loss) + 1)

    # ----------------------------------------------------------------- #
    # 1. Loss curves
    try:
        if train_loss.size and val_loss.size:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.title("SPR_BENCH Loss Curves (Sequence Classification)")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ----------------------------------------------------------------- #
    # 2. Accuracy curves
    try:
        if train_acc.size and val_acc.size:
            plt.figure()
            plt.plot(epochs, train_acc, label="Train")
            plt.plot(epochs, val_acc, label="Validation")
            plt.title("SPR_BENCH Accuracy Curves (Sequence Classification)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # ----------------------------------------------------------------- #
    # 3. Macro-F1 curves
    try:
        if train_f1.size and val_f1.size:
            plt.figure()
            plt.plot(epochs, train_f1, label="Train")
            plt.plot(epochs, val_f1, label="Validation")
            plt.title("SPR_BENCH Macro-F1 Curves (Sequence Classification)")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_f1_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ----------------------------------------------------------------- #
    # 4. Confusion matrix on test set
    try:
        preds = np.asarray(data.get("predictions", []), dtype=int)
        gts = np.asarray(data.get("ground_truth", []), dtype=int)
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1

            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title("SPR_BENCH Confusion Matrix (Test Set)")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            ticks = np.arange(num_classes)
            plt.xticks(ticks)
            plt.yticks(ticks)
            fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            print(f"Test accuracy: {(preds == gts).mean()*100:.2f}%")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
else:
    print("SPR_BENCH data not found in experiment_data.npy")

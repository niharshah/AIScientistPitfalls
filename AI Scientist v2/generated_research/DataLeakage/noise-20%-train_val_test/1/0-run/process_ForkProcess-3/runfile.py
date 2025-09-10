import matplotlib.pyplot as plt
import numpy as np
import os

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
    train_loss = np.array(data["losses"]["train_loss"])
    val_loss = np.array(data["losses"]["val_loss"])
    train_acc = np.array(data["metrics"]["train_acc"])
    val_acc = np.array(data["metrics"]["val_acc"])
    epochs = np.arange(1, len(train_loss) + 1)

    # ----------------------------------------------------------------- #
    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.title("SPR_BENCH Loss Curves (Sequence Classification)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ----------------------------------------------------------------- #
    # 2. Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.title("SPR_BENCH Accuracy Curves (Sequence Classification)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # ----------------------------------------------------------------- #
    # 3. Confusion matrix on test set
    try:
        preds = np.array(data["predictions"])
        gts = np.array(data["ground_truth"])
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
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
        plt.close()

        test_acc = (preds == gts).mean()
        print(f"Test accuracy from saved predictions: {test_acc*100:.2f}%")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
else:
    print("SPR_BENCH data not found in experiment_data.npy")

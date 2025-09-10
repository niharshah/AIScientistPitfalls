import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
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

# --------------------------------------------------------------------- #
if "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    # safely get arrays
    train_loss = np.array(data["losses"].get("train", []))
    val_loss = np.array(data["losses"].get("val", []))
    train_f1 = np.array(data["metrics"].get("train", []))
    val_f1 = np.array(data["metrics"].get("val", []))
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    epochs = np.arange(1, len(train_loss) + 1)

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
            plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2. Macro-F1 curves
    try:
        if train_f1.size and val_f1.size:
            plt.figure()
            plt.plot(epochs, train_f1, label="Train")
            plt.plot(epochs, val_f1, label="Validation")
            plt.title("SPR_BENCH Macro-F1 Curves (Sequence Classification)")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "spr_bench_f1_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # 3. Confusion matrix
    try:
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title("SPR_BENCH Confusion Matrix (Test Set)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            ticks = np.arange(num_classes)
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            plt.close()
            print(f"Test accuracy: {(preds == gts).mean()*100:.2f}%")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
else:
    print("SPR_BENCH results not found in experiment_data.npy")

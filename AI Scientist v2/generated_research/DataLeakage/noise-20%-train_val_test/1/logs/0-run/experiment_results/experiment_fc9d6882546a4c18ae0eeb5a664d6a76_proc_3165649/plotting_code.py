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

    # extract arrays safely
    train_loss = np.asarray(data["losses"].get("train", []), dtype=float)
    val_loss = np.asarray(data["losses"].get("val", []), dtype=float)
    train_acc = np.asarray(
        [m["acc"] for m in data["metrics"].get("train", [])], dtype=float
    )
    val_acc = np.asarray(
        [m["acc"] for m in data["metrics"].get("val", [])], dtype=float
    )
    train_f1 = np.asarray(
        [m["f1"] for m in data["metrics"].get("train", [])], dtype=float
    )
    val_f1 = np.asarray([m["f1"] for m in data["metrics"].get("val", [])], dtype=float)
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
    # 2. Accuracy & F1 curves
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.plot(epochs, train_f1, "--", label="Train F1")
        plt.plot(epochs, val_f1, "--", label="Val F1")
        plt.title("SPR_BENCH Accuracy & Macro-F1 (Sequence Classification)")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_acc_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating acc/F1 curve: {e}")
        plt.close()

    # ----------------------------------------------------------------- #
    # 3. Confusion matrix on test set
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
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            plt.close()

            test_acc = (preds == gts).mean()
            # macro-F1
            f1_scores = []
            for c in range(num_classes):
                tp = cm[c, c]
                fp = cm[:, c].sum() - tp
                fn = cm[c, :].sum() - tp
                precision = tp / (tp + fp) if tp + fp else 0
                recall = tp / (tp + fn) if tp + fn else 0
                f1_scores.append(
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0
                )
            test_f1 = np.mean(f1_scores)
            print(f"Test Acc: {test_acc*100:.2f}% | Test Macro-F1: {test_f1:.4f}")
        else:
            print("Predictions / ground truth not found for confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
else:
    print("SPR_BENCH data not found in experiment_data.npy")

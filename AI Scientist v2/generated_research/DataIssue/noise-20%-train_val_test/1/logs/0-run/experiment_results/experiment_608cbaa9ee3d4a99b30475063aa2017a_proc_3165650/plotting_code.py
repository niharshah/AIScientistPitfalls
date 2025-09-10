import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------- setup & load --------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Proceed only if SPR_BENCH exists
if "SPR_BENCH" in experiment_data:
    d = experiment_data["SPR_BENCH"]
    train_loss = np.asarray(d["losses"]["train"])
    val_loss = np.asarray(d["losses"]["val"])
    train_acc = np.asarray(d["metrics"]["train_acc"])
    val_acc = np.asarray(d["metrics"]["val_acc"])
    train_f1 = np.asarray(d["metrics"]["train_f1"])
    val_f1 = np.asarray(d["metrics"]["val_f1"])
    epochs = np.arange(1, len(train_loss) + 1)

    # --------------------------- 1. Loss curves ----------------------------- #
    try:
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
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ------------------------- 2. Accuracy curves --------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.title("SPR_BENCH Accuracy Curves (Sequence Classification)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

    # --------------------------- 3. F1 curves ------------------------------ #
    try:
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
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # ------------------- 4. Confusion matrix (test set) --------------------- #
    try:
        preds = np.asarray(d["predictions"])
        gts = np.asarray(d["ground_truth"])
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
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------------------- 5. Per-class F1 bar plot ------------------------ #
    try:
        per_class_f1 = []
        for c in range(num_classes):
            tp = np.sum((preds == c) & (gts == c))
            fp = np.sum((preds == c) & (gts != c))
            fn = np.sum((preds != c) & (gts == c))
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            per_class_f1.append(f1)

        plt.figure()
        plt.bar(np.arange(num_classes), per_class_f1)
        plt.title("SPR_BENCH Per-Class F1 (Test Set)")
        plt.xlabel("Class")
        plt.ylabel("F1 Score")
        plt.savefig(os.path.join(working_dir, "spr_bench_per_class_f1.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating per-class F1 plot: {e}")
        plt.close()

    # --------------------------- Print metrics ----------------------------- #
    test_acc = (preds == gts).mean()
    macro_f1 = per_class_f1 and float(np.mean(per_class_f1))
    print(f"Test accuracy: {test_acc*100:.2f}%  |  Test macro-F1: {macro_f1:.4f}")
else:
    print("SPR_BENCH data not found in experiment_data.npy")

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# Setup & data loading
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:
    train_loss = spr["losses"]["train"]
    val_loss = spr["losses"]["val"]
    train_metrics = spr["metrics"]["train"]  # list of dicts
    val_metrics = spr["metrics"]["val"]
    preds = np.array(spr["predictions"])
    gts = np.array(spr["ground_truth"])
    epochs = np.arange(1, len(train_loss) + 1)

    # ------------------------------------------------------------------ #
    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, marker="o", label="Train")
        plt.plot(epochs, val_loss, marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves (Classification Task)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 2. Accuracy curves
    try:
        plt.figure()
        tr_acc = [m["acc"] for m in train_metrics]
        val_acc = [m["acc"] for m in val_metrics]
        plt.plot(epochs, tr_acc, marker="o", label="Train")
        plt.plot(epochs, val_acc, marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves (Classification Task)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 3. F1 curves
    try:
        plt.figure()
        tr_f1 = [m["f1"] for m in train_metrics]
        val_f1 = [m["f1"] for m in val_metrics]
        plt.plot(epochs, tr_f1, marker="o", label="Train")
        plt.plot(epochs, val_f1, marker="x", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH F1 Curves (Classification Task)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 4. Confusion matrix
    try:
        num_classes = len(np.unique(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (Test Set)")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Evaluation metrics
    try:
        test_acc = np.mean(preds == gts)
        # Simple macro-F1 (two classes common, but we generalise)
        from sklearn.metrics import f1_score

        test_f1 = f1_score(gts, preds, average="macro")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Macro-F1: {test_f1:.4f}")
    except Exception as e:
        print(f"Error computing test metrics: {e}")

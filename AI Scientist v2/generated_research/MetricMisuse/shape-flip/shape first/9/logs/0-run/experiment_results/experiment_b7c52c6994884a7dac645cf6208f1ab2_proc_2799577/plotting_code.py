import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    val_swa = data["metrics"]["val_swa"]
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])

    # 1. loss curves
    try:
        plt.figure(figsize=(5, 3))
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, "--", label="Val")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. validation SWA curves
    try:
        plt.figure(figsize=(5, 3))
        epochs = np.arange(1, len(val_swa) + 1)
        plt.plot(epochs, val_swa, color="green")
        plt.title("SPR_BENCH Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_swa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve plot: {e}")
        plt.close()

    # 3. confusion matrix for test predictions
    try:
        if preds.size and gts.size:
            classes = np.unique(np.concatenate([gts, preds]))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, shrink=0.8)
            plt.title("SPR_BENCH Confusion Matrix\nRows: GT, Cols: Pred")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.xticks(classes)
            plt.yticks(classes)
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # evaluation summary
    try:
        test_acc = (preds == gts).mean() if preds.size else float("nan")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        if val_swa:
            print(f"Last Validation SWA: {val_swa[-1]:.4f}")
    except Exception as e:
        print(f"Error computing summary metrics: {e}")

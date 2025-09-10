import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["NoFeedForwardLayer"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    epochs = range(1, len(exp["losses"]["train"]) + 1)

    # -------- plot loss curves --------
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train")
        plt.plot(epochs, exp["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves – SPR_BENCH\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------- plot MCC curves --------
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["train_MCC"], label="Train")
        plt.plot(epochs, exp["metrics"]["val_MCC"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("MCC Curves – SPR_BENCH\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve plot: {e}")
        plt.close()

    # -------- confusion matrix --------
    try:
        y_true = np.array(exp["ground_truth"])
        y_pred = np.array(exp["predictions"])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix – SPR_BENCH\nLeft: Ground Truth, Right: Predicted")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # -------- print final metrics --------
    test_mcc = matthews_corrcoef(exp["ground_truth"], exp["predictions"])
    test_f1 = f1_score(exp["ground_truth"], exp["predictions"], average="macro")
    print(f"Test MCC: {test_mcc:.4f} | Test F1: {test_f1:.4f}")

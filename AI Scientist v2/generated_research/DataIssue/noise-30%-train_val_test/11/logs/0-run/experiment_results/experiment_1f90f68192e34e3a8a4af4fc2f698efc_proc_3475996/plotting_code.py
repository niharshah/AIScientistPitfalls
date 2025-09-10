import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------- load experiment data -----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    rec = experiment_data["NoRelVec"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    rec = None

if rec is not None:
    epochs = np.array(rec["epochs"])
    train_loss = np.array(rec["losses"]["train"])
    val_loss = np.array(rec["losses"]["val"])
    train_f1 = np.array(rec["metrics"]["train_macro_f1"])
    val_f1 = np.array(rec["metrics"]["val_macro_f1"])
    preds = np.array(rec["predictions"])
    trues = np.array(rec["ground_truth"])

    # ----------------------- loss curve -----------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Loss Curve\nLeft: Train, Right: Validation")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ----------------------- F1 curve -----------------------
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH – Macro-F1 Curve\nLeft: Train, Right: Validation")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ----------------------- confusion matrix -----------------------
    try:
        # build confusion matrix with numpy
        cm = np.zeros((len(np.unique(trues)), len(np.unique(trues))), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=8,
                )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH – Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ----------------------- summary metric print -----------------------
    print(f"Test Macro-F1: {rec.get('test_macro_f1', 'N/A')}")

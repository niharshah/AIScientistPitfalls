import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data.get("SPR_BENCH", None)
    if data is None:
        raise KeyError("SPR_BENCH split not found in experiment_data")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

# ---------------- compute & print metric ---------------
if data is not None:
    try:
        from sklearn.metrics import f1_score

        y_true = np.array(data["ground_truth"])
        y_pred = np.array(data["predictions"])
        test_macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Test Macro-F1 from stored predictions: {test_macro_f1:.4f}")
    except Exception as e:
        print(f"Error computing test macro-F1: {e}")

# ---------------- plotting section --------------------
if data is not None:
    epochs = np.array(data["epochs"])

    # 1. Loss curves ---------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. Validation macro-F1 -------------------------------------------------
    try:
        val_f1 = np.array(data["metrics"]["val_macro_f1"])
        plt.figure()
        plt.plot(epochs, val_f1, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.ylim(0, 1.0)
        plt.title("SPR_BENCH Validation Macro-F1 Across Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_val_f1.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation F1 plot: {e}")
        plt.close()

    # 3. Confusion matrix ----------------------------------------------------
    try:
        labels = np.unique(y_true)
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.xticks(labels)
        plt.yticks(labels)
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

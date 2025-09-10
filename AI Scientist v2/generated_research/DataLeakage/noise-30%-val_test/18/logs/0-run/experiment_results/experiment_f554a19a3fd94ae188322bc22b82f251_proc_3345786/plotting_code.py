import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- setup -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    try:
        ref = experiment_data["NoDropout"]["SPR_BENCH"]
        train_loss = ref["losses"]["train"]
        val_loss = ref["losses"]["val"]
        train_mcc = ref["metrics"]["train_MCC"]
        val_mcc = ref["metrics"]["val_MCC"]
        preds = np.array(ref["predictions"])
        gts = np.array(ref["ground_truth"])
    except Exception as e:
        print(f"Error extracting data: {e}")
        experiment_data = None

# ---------------- plotting -----------------
if experiment_data:
    # 1. Loss curves
    try:
        plt.figure()
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. MCC curves
    try:
        plt.figure()
        epochs = np.arange(1, len(train_mcc) + 1)
        plt.plot(epochs, train_mcc, label="Train MCC")
        plt.plot(epochs, val_mcc, label="Val MCC")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("SPR_BENCH: Training vs Validation MCC")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot: {e}")
        plt.close()

    # 3. Confusion matrix heatmap
    try:
        from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score

        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------------- metrics -----------------
    try:
        from sklearn.metrics import matthews_corrcoef, f1_score

        test_mcc = matthews_corrcoef(gts, preds)
        test_f1 = f1_score(gts, preds, average="macro")
        print(f"Test MCC = {test_mcc:.4f} | Test F1 = {test_f1:.4f}")
    except Exception as e:
        print(f"Error computing metrics: {e}")

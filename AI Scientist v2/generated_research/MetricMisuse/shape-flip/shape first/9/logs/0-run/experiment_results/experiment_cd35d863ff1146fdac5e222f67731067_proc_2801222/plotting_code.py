import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------  set up paths  -------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------  load experiment data  ----------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    try:
        ed = experiment_data["NoRNN_MeanPool"]["SPR_BENCH"]
        train_loss = np.asarray(ed["losses"]["train"])
        val_loss = np.asarray(ed["losses"]["val"])
        val_swa = np.asarray(ed["metrics"]["val"])
        preds = np.asarray(ed["predictions"])
        gts = np.asarray(ed["ground_truth"])
        test_swa = ed["metrics"]["test"]
    except Exception as e:
        print(f"Error extracting data from experiment dictionary: {e}")
        ed = None

if ed:
    epochs = np.arange(1, len(train_loss) + 1)

    # -------- 1. Loss curves --------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Training vs. Validation Loss — SPR_BENCH")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------- 2. Validation metric curve --------------------------------
    try:
        plt.figure()
        plt.plot(epochs, val_swa, marker="o", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("Validation Shape-Weighted Accuracy — SPR_BENCH")
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve plot: {e}")
        plt.close()

    # -------- 3. Confusion matrix ---------------------------------------
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix — SPR_BENCH (Test Set)")
        plt.colorbar(im)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # -------- 4. Print evaluation metric --------------------------------
    print(f"Test Shape-Weighted Accuracy: {test_swa:.4f}")

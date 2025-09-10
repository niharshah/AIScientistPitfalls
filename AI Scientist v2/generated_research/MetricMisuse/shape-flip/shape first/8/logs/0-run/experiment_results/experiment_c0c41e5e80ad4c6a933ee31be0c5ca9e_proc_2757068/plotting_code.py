import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["no_histogram"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

# -------- plotting --------
if ed is not None:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Loss over Epochs\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) SWA curves
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_swa"], label="Train SWA")
        plt.plot(epochs, ed["metrics"]["val_swa"], label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(
            "SPR_BENCH – Shape-Weighted Accuracy over Epochs\nTrain vs Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # 3) Confusion matrix on test set
    try:
        y_true = np.array(ed["ground_truth"])
        y_pred = np.array(ed["predictions"])
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.colorbar()
        plt.title(
            "SPR_BENCH – Confusion Matrix\nLeft: Ground Truth classes, Right: Predicted classes"
        )
        fname = os.path.join(working_dir, "spr_bench_test_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

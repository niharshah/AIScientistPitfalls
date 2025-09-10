import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["binarized_hist"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) SWA curves
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_swa"], label="Train")
        plt.plot(epochs, ed["metrics"]["val_swa"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("spr_bench – Training vs Validation SWA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_swa_curve.png")
        plt.savefig(fname)
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
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("spr_bench – Confusion Matrix (Test Set)")
        plt.colorbar(label="Count")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # Print evaluation metrics
    test_swa = ed["metrics"].get("test_swa", None)
    if test_swa is not None:
        print(f"Test Shape-Weighted Accuracy (SWA): {test_swa:.3f}")
        print("Confusion Matrix:\n", cm)

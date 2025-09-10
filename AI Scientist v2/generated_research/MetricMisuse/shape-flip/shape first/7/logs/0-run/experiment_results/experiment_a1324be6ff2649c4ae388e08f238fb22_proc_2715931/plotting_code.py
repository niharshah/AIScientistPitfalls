import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------
# Paths and data loading
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Only proceed if data present
if "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    # -------------------------- Metric computation ----------
    y_true = np.array(data.get("ground_truth", []))
    y_pred = np.array(data.get("predictions", []))
    test_acc = float((y_true == y_pred).mean() if len(y_true) else np.nan)
    print(f"Test set accuracy: {test_acc:.4f}")

    # -------------------------- PLOT 1: Loss curves ---------
    try:
        plt.figure()
        tr = np.array(data["losses"]["train"])
        val = np.array(data["losses"]["val"])
        plt.plot(tr[:, 0], tr[:, 1], label="Train")
        plt.plot(val[:, 0], val[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------------------------- PLOT 2: HWA curves ----------
    try:
        plt.figure()
        tr = np.array(data["metrics"]["train"])
        val = np.array(data["metrics"]["val"])
        plt.plot(tr[:, 0], tr[:, 1], label="Train")
        plt.plot(val[:, 0], val[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic-Weighted Accuracy")
        plt.title("SPR_BENCH HWA Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve: {e}")
        plt.close()

    # -------------------------- PLOT 3: Confusion matrix ----
    try:
        if len(y_true):
            plt.figure()
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1
            im = plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(
                        j, i, str(cm[i, j]), ha="center", va="center", color="black"
                    )
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
else:
    print("No SPR_BENCH data available in experiment_data.")

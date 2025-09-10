import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths / load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper --------------------------------------------------------
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------- per-dataset plots --------------------------------------------
all_test_swa = {}
for key, log in experiment_data.items():
    epochs = log.get("epochs", [])

    # 1) Loss curves -------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, log["losses"]["train"], label="train")
        plt.plot(epochs, log["losses"]["dev"], label="dev")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{key} – Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{key}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {key}: {e}")
        plt.close()

    # 2) SWA curves --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, log["metrics"]["train_SWA"], label="train_SWA")
        plt.plot(epochs, log["metrics"]["dev_SWA"], label="dev_SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{key} – SWA Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{key}_SWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve for {key}: {e}")
        plt.close()

    # 3) Confusion matrix --------------------------------------------------
    try:
        y_true = np.asarray(log.get("ground_truth", []))
        y_pred = np.asarray(log.get("predictions", []))
        if y_true.size and y_pred.size:
            n_classes = int(max(y_true.max(), y_pred.max())) + 1
            cm = confusion_matrix(y_true, y_pred, n_classes)
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{key} – Confusion Matrix")
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            fname = os.path.join(working_dir, f"{key}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        else:
            raise ValueError("Missing ground_truth or predictions.")
    except Exception as e:
        print(f"Error creating confusion matrix for {key}: {e}")
        plt.close()

    # collect test SWA for comparison plot
    test_swa = log.get("test_SWA", None)
    if test_swa is not None:
        all_test_swa[key] = test_swa

# ---------- comparison plot across datasets -------------------------------
if len(all_test_swa) > 1:
    try:
        plt.figure()
        keys, vals = zip(*all_test_swa.items())
        plt.bar(keys, vals, color="tab:purple")
        plt.ylim(0, 1)
        plt.title("Test SWA Comparison Across Datasets")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "comparison_test_SWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()

print("Plotting complete; figures saved to", working_dir)

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths / load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper: confusion matrix
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------- per-dataset plots
test_swa_dict = {}
for ds_key, log in experiment_data.items():
    epochs = log.get("epochs", [])
    # 1) loss curves
    try:
        plt.figure()
        plt.plot(epochs, log["losses"]["train"], label="Train")
        plt.plot(epochs, log["losses"]["dev"], label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_key} Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_key}: {e}")
        plt.close()

    # 2) dev SWA curve
    try:
        plt.figure()
        plt.plot(epochs, log["metrics"]["dev_SWA"], color="tab:green")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Dev SWA")
        plt.title(f"{ds_key} Dev Shape-Weighted-Accuracy")
        fname = os.path.join(working_dir, f"{ds_key}_dev_SWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating dev_SWA curve for {ds_key}: {e}")
        plt.close()

    # 3) confusion matrix
    try:
        y_true = np.asarray(log.get("ground_truth", []))
        y_pred = np.asarray(log.get("predictions", []))
        if y_true.size and y_pred.size:
            n_cls = max(y_true.max(), y_pred.max()) + 1
            cm = confusion_matrix(y_true, y_pred, n_cls)
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(f"{ds_key} Confusion Matrix (Test)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            fname = os.path.join(working_dir, f"{ds_key}_confusion_matrix.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_key}: {e}")
        plt.close()

    # collect final test SWA for overall comparison
    if "test_SWA" in log:
        test_swa_dict[ds_key] = log["test_SWA"]

# ---------- comparison bar plot of test SWA across datasets
try:
    if test_swa_dict:
        plt.figure()
        keys, vals = zip(*test_swa_dict.items())
        plt.bar(keys, vals, color="tab:orange")
        plt.ylim(0, 1)
        plt.ylabel("Test Shape-Weighted-Accuracy")
        plt.title("Test SWA Comparison Across Datasets")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, "all_datasets_test_SWA.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA comparison plot: {e}")
    plt.close()

# ---------- print key metrics
for k, v in test_swa_dict.items():
    print(f"{k}: Test SWA = {v:.4f}")

print("Plotting complete; figures saved to", working_dir)

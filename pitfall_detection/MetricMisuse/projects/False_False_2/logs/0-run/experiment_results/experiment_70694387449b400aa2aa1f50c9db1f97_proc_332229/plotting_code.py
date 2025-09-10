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


# ---------------- helper
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


test_metric_summary = {}  # collect for cross-dataset comparison

# ---------------- per-dataset plots
for dataset_key, log in experiment_data.items():
    epochs = log.get("epochs", [])

    # 1) Loss curves ---------------------------------------------------------
    try:
        if epochs and log["losses"]:
            plt.figure()
            plt.plot(epochs, log["losses"]["train"], label="train")
            plt.plot(epochs, log["losses"]["dev"], label="dev")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset_key} Loss Curve")
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset_key}_loss_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dataset_key}: {e}")
        plt.close()

    # 2) SWA metric curve ----------------------------------------------------
    try:
        if epochs and "dev_SWA" in log.get("metrics", {}):
            plt.figure()
            plt.plot(
                epochs, log["metrics"]["dev_SWA"], label="dev_SWA", color="tab:green"
            )
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.title(f"{dataset_key} Shape-Weighted Accuracy Curve")
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset_key}_swa_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve for {dataset_key}: {e}")
        plt.close()

    # 3) Test metric bar plot ------------------------------------------------
    try:
        test_swa = log.get("test_SWA", None)
        if test_swa is not None:
            test_metric_summary[dataset_key] = test_swa
            plt.figure()
            plt.bar(["test_SWA"], [test_swa], color="tab:blue")
            plt.ylim(0, 1)
            plt.title(f"{dataset_key} Test Metrics")
            plt.text(0, test_swa + 0.02, f"{test_swa:.2f}", ha="center")
            fname = os.path.join(working_dir, f"{dataset_key}_test_metrics.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot for {dataset_key}: {e}")
        plt.close()

    # 4) Confusion matrix ----------------------------------------------------
    try:
        y_true = np.asarray(log.get("ground_truth", []))
        y_pred = np.asarray(log.get("predictions", []))
        if y_true.size and y_pred.size:
            n_classes = max(y_true.max(), y_pred.max()) + 1
            cm = confusion_matrix(y_true, y_pred, n_classes)
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dataset_key} Confusion Matrix")
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
            fname = os.path.join(working_dir, f"{dataset_key}_confusion_matrix.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dataset_key}: {e}")
        plt.close()

# ---------------- comparison plot across datasets ---------------------------
try:
    if test_metric_summary:
        plt.figure()
        ds_names, ds_scores = zip(*test_metric_summary.items())
        plt.bar(ds_names, ds_scores, color="tab:orange")
        plt.ylim(0, 1)
        plt.title("Test SWA Comparison Across Datasets")
        for i, v in enumerate(ds_scores):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "comparison_test_SWA.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()

print("Plotting complete; figures saved to", working_dir)

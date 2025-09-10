import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------------
# Identify model key and dataset key automatically
if experiment_data:
    model_key = next(iter(experiment_data))
    dataset_key = next(iter(experiment_data[model_key]))
    store = experiment_data[model_key][dataset_key]
else:
    store = None

if store is None:
    print("No experiment data found — nothing to plot.")
else:
    epochs = store.get("epochs", [])
    losses_tr = store["losses"].get("train", [])
    losses_val = store["losses"].get("val", [])
    metrics = store["metrics"]
    test_metrics = store.get("test_metrics", {})
    y_pred = np.array(store.get("predictions", []))
    y_true = np.array(store.get("ground_truth", []))

    # 1) Loss curve -------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset_key}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # Helper to plot any metric ------------------------------------------
    def plot_metric(metric_name, ylabel):
        try:
            plt.figure()
            plt.plot(epochs, metrics["train"][metric_name], label="Train")
            plt.plot(epochs, metrics["val"][metric_name], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(f"{dataset_key}: {metric_name} over Epochs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset_key}_{metric_name}_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating {metric_name} curve: {e}")
            plt.close()

    # 2) CmpWA curve
    plot_metric("CmpWA", "Complexity-Weighted Accuracy")

    # 3) CWA curve
    plot_metric("CWA", "Color-Weighted Accuracy")

    # 4) SWA curve
    plot_metric("SWA", "Shape-Weighted Accuracy")

    # 5) Confusion Matrix -------------------------------------------------
    try:
        if y_pred.size and y_true.size:
            num_cls = max(y_true.max(), y_pred.max()) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dataset_key}: Confusion Matrix")
            for i in range(num_cls):
                for j in range(num_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{dataset_key}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        else:
            print("Skipped confusion matrix: no prediction data found.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------------------------------------------------------------------
    # Print stored test metrics
    if test_metrics:
        print("Test-set metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

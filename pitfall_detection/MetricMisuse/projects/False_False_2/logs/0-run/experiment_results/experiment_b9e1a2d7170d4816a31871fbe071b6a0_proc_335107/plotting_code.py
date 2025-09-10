import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------------------- iterate and plot
for exp_name, ds_dict in experiment_data.items():
    for dataset_name, logs in ds_dict.items():
        epochs = logs.get("epochs", [])
        # -------- 1) Loss curves
        try:
            plt.figure()
            plt.plot(epochs, logs["losses"]["train"], label="Train")
            plt.plot(epochs, logs["losses"]["dev"], label="Dev")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset_name}: Loss Curve (Train vs Dev)")
            plt.legend()
            fname = f"{dataset_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dataset_name}: {e}")
            plt.close()

        # -------- 2) PHA curves
        try:
            plt.figure()
            plt.plot(epochs, logs["metrics"]["train_PHA"], label="Train PHA")
            plt.plot(epochs, logs["metrics"]["dev_PHA"], label="Dev PHA")
            plt.xlabel("Epoch")
            plt.ylabel("PHA")
            plt.title(f"{dataset_name}: PHA Curve (Train vs Dev)")
            plt.legend()
            fname = f"{dataset_name}_pha_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating PHA curve for {dataset_name}: {e}")
            plt.close()

        # -------- 3) Confusion matrix
        try:
            y_true = np.asarray(logs["ground_truth"])
            y_pred = np.asarray(logs["predictions"])
            n_cls = int(max(y_true.max(), y_pred.max()) + 1)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
            plt.figure()
            im = plt.imshow(cm_norm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dataset_name}: Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            fname = f"{dataset_name}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dataset_name}: {e}")
            plt.close()

        # -------- 4) Test metric bar chart
        try:
            metrics = logs.get("test_metrics", {})
            labels = list(metrics.keys())
            values = [metrics[k] for k in labels]
            plt.figure()
            plt.bar(labels, values, color=["green", "orange", "red"])
            plt.ylim(0, 1)
            plt.title(f"{dataset_name}: Test Metrics (SWA/CWA/PHA)")
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = f"{dataset_name}_test_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating test metric bar chart for {dataset_name}: {e}")
            plt.close()

        # -------- print metrics
        if "test_metrics" in logs:
            print(f"{exp_name}-{dataset_name} Test metrics:", logs["test_metrics"])

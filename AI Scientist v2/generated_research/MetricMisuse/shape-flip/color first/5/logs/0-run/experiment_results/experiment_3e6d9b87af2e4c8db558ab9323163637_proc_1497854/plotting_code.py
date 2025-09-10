import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------
for model_name, model_dict in experiment_data.items():
    for dataset_name, d in model_dict.items():
        losses = d["losses"]
        metrics = d["metrics"]
        test_metrics = d["test_metrics"]

        # 1) Loss curve ----------------------------------------------------
        try:
            plt.figure()
            epochs = range(1, len(losses["train"]) + 1)
            plt.plot(epochs, losses["train"], label="train")
            plt.plot(epochs, losses["val"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset_name} | {model_name} - Training vs Validation Loss")
            plt.legend()
            fname = f"{dataset_name}_{model_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # Helper to plot metric histories ---------------------------------
        def plot_metric(metric_key, full_name):
            try:
                plt.figure()
                plt.plot(epochs, metrics["train"][metric_key], label="train")
                plt.plot(epochs, metrics["val"][metric_key], label="val")
                plt.xlabel("Epoch")
                plt.ylabel(full_name)
                plt.title(f"{dataset_name} | {model_name} - {full_name} over Epochs")
                plt.legend()
                fname = f"{dataset_name}_{model_name}_{metric_key}_curve.png"
                plt.savefig(os.path.join(working_dir, fname), dpi=150)
                plt.close()
            except Exception as e_inner:
                print(f"Error creating {metric_key} curve: {e_inner}")
                plt.close()

        # 2) CWA curve
        plot_metric("CWA", "Color-Weighted Accuracy")
        # 3) SWA curve
        plot_metric("SWA", "Shape-Weighted Accuracy")
        # 4) CmpWA curve
        plot_metric("CmpWA", "Complexity-Weighted Accuracy")

        # 5) Test metrics bar chart ---------------------------------------
        try:
            plt.figure()
            bars = ["CWA", "SWA", "CmpWA"]
            values = [test_metrics["CWA"], test_metrics["SWA"], test_metrics["CmpWA"]]
            plt.bar(bars, values, color=["tab:blue", "tab:orange", "tab:green"])
            plt.ylim(0, 1)
            plt.title(f"{dataset_name} | {model_name} - Test Weighted Accuracies")
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = f"{dataset_name}_{model_name}_test_metrics.png"
            plt.savefig(os.path.join(working_dir, fname), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating test metrics bar chart: {e}")
            plt.close()

        # Print final test metrics ----------------------------------------
        print(
            f"{dataset_name} | {model_name} TEST -> "
            f'Loss: {test_metrics["loss"]:.4f}, '
            f'CWA: {test_metrics["CWA"]:.4f}, '
            f'SWA: {test_metrics["SWA"]:.4f}, '
            f'CmpWA: {test_metrics["CmpWA"]:.4f}'
        )

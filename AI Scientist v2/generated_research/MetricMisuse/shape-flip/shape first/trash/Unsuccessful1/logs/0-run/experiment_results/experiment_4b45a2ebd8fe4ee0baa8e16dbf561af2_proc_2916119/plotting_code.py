import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data.get("SPR_BENCH", None)
    if data is None:
        raise ValueError("SPR_BENCH key not found.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    epochs = np.arange(1, len(data["losses"]["train"]) + 1)
    # ----------- Plot 1: Loss curves ---------------
    try:
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------- Plot 2: Validation metrics curves ---------------
    try:
        swa = [m["SWA"] for m in data["metrics"]["val"]]
        cwa = [m["CWA"] for m in data["metrics"]["val"]]
        hwa = [m["HWA"] for m in data["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, hwa, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(
            "SPR_BENCH Validation Weighted Accuracies\nLeft: SWA, Mid: CWA, Right: HWA"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metric_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve plot: {e}")
        plt.close()

    # ----------- Plot 3: Test metrics bar chart ---------------
    try:
        test_metrics = data["metrics"]["test"]
        labels = list(test_metrics.keys())
        values = [test_metrics[k] for k in labels]
        plt.figure()
        plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Test Metrics\nBar chart of SWA, CWA, HWA")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar plot: {e}")
        plt.close()

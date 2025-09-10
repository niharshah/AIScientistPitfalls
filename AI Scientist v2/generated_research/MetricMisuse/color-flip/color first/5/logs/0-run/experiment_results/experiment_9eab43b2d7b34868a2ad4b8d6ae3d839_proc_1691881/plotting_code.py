import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# shortcut to dataset dict
exp = experiment_data.get("TwoClusterGranularity", {}).get("SPR_BENCH", {})

losses = exp.get("losses", {})
metrics_val = exp.get("metrics", {}).get("val", [])
metrics_test = exp.get("metrics", {}).get("test", {})

# ----- Plot 1: loss curves ----------------------------------------------------
try:
    epochs = range(1, len(losses.get("train", [])) + 1)
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train Loss")
    plt.plot(epochs, losses.get("val", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ----- Plot 2: validation metrics --------------------------------------------
try:
    if metrics_val:
        epochs = [m["epoch"] for m in metrics_val]
        acc = [m["acc"] for m in metrics_val]
        cwa = [m["cwa"] for m in metrics_val]
        swa = [m["swa"] for m in metrics_val]
        ccwa = [m["ccwa"] for m in metrics_val]

        plt.figure()
        plt.plot(epochs, acc, label="ACC")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, ccwa, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Validation Metrics Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating validation metric curve: {e}")
    plt.close()

# ----- Plot 3: test metrics ---------------------------------------------------
try:
    if metrics_test:
        names = ["ACC", "CWA", "SWA", "CCWA"]
        values = [
            metrics_test.get("acc", 0),
            metrics_test.get("cwa", 0),
            metrics_test.get("swa", 0),
            metrics_test.get("ccwa", 0),
        ]
        plt.figure()
        plt.bar(names, values, color="skyblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()

# ----- Print test metrics -----------------------------------------------------
if metrics_test:
    print("FINAL TEST METRICS:")
    for k, v in metrics_test.items():
        print(f"  {k.upper():4s}: {v:.3f}")

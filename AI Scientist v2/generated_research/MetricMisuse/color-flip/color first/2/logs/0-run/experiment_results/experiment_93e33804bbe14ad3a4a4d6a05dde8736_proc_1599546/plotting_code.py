import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_key = "SPR_BENCH"
ed = experiment_data.get(data_key, {})

loss_train = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])
val_metrics = ed.get("metrics", {}).get("val", [])
test_metrics = ed.get("metrics", {}).get("test", {})

epochs = list(range(1, len(loss_train) + 1))

# ------------------ Plot 1: loss curves ------------------ #
try:
    plt.figure()
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{data_key} Loss Curves\nLeft: Training, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{data_key}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------ Plot 2: validation metric curves ------------------ #
try:
    if val_metrics:
        cwa = [m["CWA"] for m in val_metrics]
        swa = [m["SWA"] for m in val_metrics]
        gcwa = [m["GCWA"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, gcwa, label="GCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{data_key} Validation Metrics over Epochs\nCWA, SWA, GCWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{data_key}_val_metrics_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metric curve: {e}")
    plt.close()

# ------------------ Plot 3: test metric bar chart ------------------ #
try:
    if test_metrics:
        labels = ["CWA", "SWA", "GCWA"]
        values = [test_metrics.get(k, 0) for k in labels]
        plt.figure()
        plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"{data_key} Test Metrics\nBar Chart of Final Scores")
        fname = os.path.join(working_dir, f"{data_key}_test_metrics_bar.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()

# ------------------ print final metrics ------------------ #
if test_metrics:
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.3f}")

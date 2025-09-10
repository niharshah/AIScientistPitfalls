import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

saved_files = []
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----- per-run accuracy / URA curves -----
for run_key, run_data in experiment_data.get("EPOCHS", {}).items():
    try:
        epochs = list(range(1, len(run_data["metrics"]["train_acc"]) + 1))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, run_data["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, run_data["metrics"]["val_acc"], label="Val Acc")
        plt.plot(epochs, run_data["metrics"]["val_ura"], label="Val URA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(f"SPR_BENCH – {run_key} Accuracy/URA Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{run_key}_acc_curves.png")
        plt.savefig(fname)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {run_key}: {e}")
        plt.close()

# ----- final test accuracy / URA bar plot -----
try:
    run_names = []
    test_accs = []
    test_uras = []
    for rk, rd in experiment_data.get("EPOCHS", {}).items():
        run_names.append(rk)
        test_accs.append(rd.get("test_acc", 0))
        test_uras.append(rd.get("test_ura", 0))
    x = np.arange(len(run_names))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, test_accs, width, label="Test Acc")
    plt.bar(x + width / 2, test_uras, width, label="Test URA")
    plt.xticks(x, run_names, rotation=45)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("SPR_BENCH – Test Accuracy vs URA by Epoch Setting")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(fname)
    saved_files.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()

print("Saved plots:", saved_files)

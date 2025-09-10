import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_key = "SPR_BENCH"
if data_key not in experiment_data:
    print(f"No data found for key {data_key}")
    exit()

runs = experiment_data[data_key]
train_losses_runs = runs["losses"]["train"]
val_losses_runs = runs["losses"]["val"]
val_metrics_runs = runs["metrics"]["val"]
test_metrics_runs = runs["metrics"]["test"]

# ---------- 1) train / val loss curves ----------
try:
    plt.figure()
    for idx, (tr, vl) in enumerate(zip(train_losses_runs, val_losses_runs)):
        epochs = np.arange(1, len(tr) + 1)
        plt.plot(epochs, tr, "-o", label=f"run{idx+1} train")
        plt.plot(epochs, vl, "--o", label=f"run{idx+1} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- 2) GCWA validation curves ----------
try:
    plt.figure()
    for idx, val_hist in enumerate(val_metrics_runs):
        gcwa = [m["GCWA"] for m in val_hist]
        epochs = np.arange(1, len(gcwa) + 1)
        plt.plot(epochs, gcwa, "-o", label=f"run{idx+1}")
    plt.xlabel("Epoch")
    plt.ylabel("GCWA")
    plt.title("SPR_BENCH: Validation GCWA over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_gcwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating GCWA curve plot: {e}")
    plt.close()

# ---------- 3) final test metrics bar chart ----------
try:
    metrics_names = ["CWA", "SWA", "GCWA"]
    x = np.arange(len(metrics_names))
    width = 0.25
    plt.figure()
    for idx, tm in enumerate(test_metrics_runs):
        vals = [tm[m] for m in metrics_names]
        plt.bar(x + width * idx, vals, width=width, label=f"run{idx+1}")
    plt.xticks(x + width, metrics_names)
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Test Metrics per Run")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar plot: {e}")
    plt.close()

print("Plots saved to", working_dir)

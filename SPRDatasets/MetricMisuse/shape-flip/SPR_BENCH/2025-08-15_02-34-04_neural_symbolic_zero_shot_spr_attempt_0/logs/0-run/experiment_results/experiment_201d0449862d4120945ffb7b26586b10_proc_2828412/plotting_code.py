import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data ------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("random_shuffle", {}).get("num_epochs", {})
if not runs:
    print("No run data found, exiting.")
    exit()

# ------------ prepare containers ------------
loss_curves = {}  # key -> dict(train=[..], val=[..])
hwa_curves = {}  # key -> list(val_hwa)
test_metrics = {}  # key -> float(test_hwa)

for run_name, run_dat in runs.items():
    loss_curves[run_name] = {
        "train": run_dat["losses"]["train"],
        "val": run_dat["losses"]["val"],
    }
    hwa_curves[run_name] = [m[2] for m in run_dat["metrics"]["val"]]
    test_metrics[run_name] = run_dat["metrics"]["test"][2]

# ------------ PLOT 1: loss curves ------------
try:
    plt.figure()
    for run, curves in loss_curves.items():
        epochs = range(1, len(curves["train"]) + 1)
        plt.plot(epochs, curves["train"], "--", label=f"{run} train")
        plt.plot(epochs, curves["val"], "-", label=f"{run} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH (Random Token Shuffle)\nTrain vs Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------ PLOT 2: HWA curves ------------
try:
    plt.figure()
    for run, vals in hwa_curves.items():
        epochs = range(1, len(vals) + 1)
        plt.plot(epochs, vals, label=f"{run}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH (Random Token Shuffle)\nValidation HWA per Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_hwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ------------ PLOT 3: test HWA bar chart ------------
try:
    plt.figure()
    keys, vals = zip(
        *sorted(test_metrics.items(), key=lambda x: int(x[0].split("_")[-1]))
    )
    plt.bar(keys, vals)
    plt.ylabel("Test HWA")
    plt.title(
        "SPR_BENCH (Random Token Shuffle)\nFinal Test HWA for Different Epoch Budgets"
    )
    plt.xticks(rotation=45)
    fname = os.path.join(working_dir, "spr_bench_test_hwa_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test HWA bar chart: {e}")
    plt.close()

# ------------ print summary table ------------
print("=== Test Metrics (HWA) ===")
for k, v in sorted(test_metrics.items(), key=lambda x: int(x[0].split("_")[-1])):
    print(f"{k:>12}: {v:.4f}")

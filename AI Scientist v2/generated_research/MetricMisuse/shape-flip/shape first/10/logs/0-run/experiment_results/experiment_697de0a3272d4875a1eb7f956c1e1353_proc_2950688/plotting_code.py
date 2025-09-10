import matplotlib.pyplot as plt
import numpy as np
import os

# --- setup & data -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# If keys are nested differently, fall back to top-level
runs = (
    experiment_data
    if "num_epochs" not in experiment_data
    else experiment_data["num_epochs"]
)

# --- 1-3 : individual loss curves --------------------------------------------
for run_name, run_data in list(runs.items())[:3]:  # safety: at most 3 plots
    try:
        tr = run_data["losses"]["train"]
        val = run_data["losses"]["val"]
        epochs = np.arange(1, len(tr) + 1)
        plt.figure()
        plt.plot(epochs, tr, label="Train")
        plt.plot(epochs, val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{run_name}: Loss Curves\nDataset: SPR_BENCH")
        plt.legend()
        fname = os.path.join(working_dir, f"{run_name}_loss_SPR_BENCH.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {run_name}: {e}")
        plt.close()

# --- 4 : aggregated test SWA --------------------------------------------------
try:
    labels, swa_vals = [], []
    for rn, rd in runs.items():
        labels.append(rn)
        swa_vals.append(rd.get("metrics", {}).get("test", 0))
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, swa_vals, color="steelblue")
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("Test SWA Comparison\nDataset: SPR_BENCH")
    fname = os.path.join(working_dir, "SWA_comparison_SPR_BENCH.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error plotting SWA comparison: {e}")
    plt.close()

# --- print final metrics ------------------------------------------------------
print("Final Test Metrics (SWA):")
for rn, rd in runs.items():
    print(f"{rn:15s}: {rd.get('metrics', {}).get('test', 0):.4f}")

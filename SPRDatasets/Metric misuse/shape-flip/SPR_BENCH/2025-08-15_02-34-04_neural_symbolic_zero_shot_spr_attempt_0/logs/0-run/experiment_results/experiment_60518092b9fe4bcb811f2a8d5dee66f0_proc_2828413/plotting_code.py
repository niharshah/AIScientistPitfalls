import matplotlib.pyplot as plt
import numpy as np
import os

# --------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("onehot_no_embedding", {})


# --------- helper for metrics extraction ----------
def get_series(run_dict, key):
    return [m[key] for m in run_dict["metrics"]["val"]]


# --------- 1. loss curves ----------
try:
    plt.figure(figsize=(6, 4))
    for run_name, run in runs.items():
        plt.plot(run["losses"]["train"], label=f"{run_name}-train")
        plt.plot(run["losses"]["val"], label=f"{run_name}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Bench: One-Hot GRU Loss Curves")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_loss_curves_all_runs.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# --------- 2. validation HWA curves ----------
try:
    plt.figure(figsize=(6, 4))
    for run_name, run in runs.items():
        hwa_vals = [m[2] for m in run["metrics"]["val"]]
        plt.plot(hwa_vals, label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.title("SPR Bench: Validation HWA Trajectories")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_val_hwa_curves_all_runs.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# --------- 3. test metrics bar chart ----------
try:
    run_names = list(runs.keys())
    swa = [runs[r]["metrics"]["test"][0] for r in run_names]
    cwa = [runs[r]["metrics"]["test"][1] for r in run_names]
    hwa = [runs[r]["metrics"]["test"][2] for r in run_names]
    x = np.arange(len(run_names))
    width = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(x - width, swa, width, label="SWA")
    plt.bar(x, cwa, width, label="CWA")
    plt.bar(x + width, hwa, width, label="HWA")
    plt.xticks(x, run_names, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("SPR Bench: Test Metrics per Run")
    plt.legend()
    fname = os.path.join(working_dir, "spr_test_metrics_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# --------- print numeric summary ----------
for name in runs:
    t = runs[name]["metrics"]["test"]
    print(f"{name}: Test SWA={t[0]:.4f}, CWA={t[1]:.4f}, HWA={t[2]:.4f}")

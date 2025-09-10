import matplotlib.pyplot as plt
import numpy as np
import os

# set working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Shortcut to the hyper-param sweep dict
runs = experiment_data.get("embed_dim", {})
dims = sorted(runs.keys())

# Collect summary for console print
summary_rows = []

# ------------------------------------------------------------------ #
# 1) Train/Dev Loss curves
try:
    plt.figure()
    for d in dims:
        ep = np.arange(1, len(runs[d]["losses"]["train"]) + 1)
        plt.plot(ep, runs[d]["losses"]["train"], label=f"train dim={d}", alpha=0.8)
        plt.plot(ep, runs[d]["losses"]["dev"], label=f"dev dim={d}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves – Synthetic SPR_BENCH (Train vs Dev)")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SyntheticSPR_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) Dev-set accuracy curves
try:
    plt.figure()
    for d in dims:
        acc = runs[d]["metrics"]["dev_acc"]
        plt.plot(np.arange(1, len(acc) + 1), acc, label=f"dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.title("Dev Accuracy – Synthetic SPR_BENCH")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SyntheticSPR_dev_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating dev accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) Final test accuracy bars
try:
    plt.figure()
    test_accs = [runs[d]["metrics"]["test"]["acc"] for d in dims]
    plt.bar(range(len(dims)), test_accs, tick_label=dims)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Embedding Dim – Synthetic SPR_BENCH")
    fname = os.path.join(working_dir, "SyntheticSPR_test_accuracy_bars.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) NRGS bars
try:
    plt.figure()
    nrg_vals = [runs[d]["metrics"]["NRGS"] for d in dims]
    plt.bar(range(len(dims)), nrg_vals, tick_label=dims, color="orange")
    plt.ylabel("NRGS")
    plt.title("NRGS by Embedding Dim – Synthetic SPR_BENCH")
    fname = os.path.join(working_dir, "SyntheticSPR_NRGS_bars.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating NRGS bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Print summary metrics
for d in dims:
    t = runs[d]["metrics"]["test"]
    row = (d, t["acc"], t["swa"], t["cwa"], runs[d]["metrics"]["NRGS"])
    summary_rows.append(row)

print("\nDim | TestAcc | SWA | CWA | NRGS")
for r in summary_rows:
    print(f"{r[0]:3} | {r[1]:7.3f} | {r[2]:4.3f} | {r[3]:4.3f} | {r[4]:4.3f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ensure working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper to grab values
hidden_runs = experiment_data.get("hidden_dim", {}).get("SPR_BENCH", {})
hidden_dims = sorted(hidden_runs.keys(), key=lambda x: int(x))

# collect metrics for printing
summary = []

# 1) Loss curves
try:
    plt.figure()
    for hd in hidden_dims:
        epochs = range(1, len(hidden_runs[hd]["losses"]["train"]) + 1)
        plt.plot(epochs, hidden_runs[hd]["losses"]["train"], label=f"h{hd}-train")
        plt.plot(
            epochs, hidden_runs[hd]["losses"]["val"], label=f"h{hd}-val", linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Validation CWA2 curves
try:
    plt.figure()
    for hd in hidden_dims:
        epochs = range(1, len(hidden_runs[hd]["metrics"]["val_cwa2"]) + 1)
        cwa = hidden_runs[hd]["metrics"]["val_cwa2"]
        plt.plot(epochs, cwa, label=f"h{hd}")
        summary.append((hd, max(cwa)))
    plt.xlabel("Epoch")
    plt.ylabel("Validation CWA2")
    plt.title("SPR_BENCH Validation CWA2 over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_CWA2_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA2 curves: {e}")
    plt.close()

# 3) Best CWA2 per hidden size
try:
    plt.figure()
    hds, best_cwa = zip(*summary) if summary else ([], [])
    plt.bar([str(h) for h in hds], best_cwa, color="skyblue")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Best Validation CWA2")
    plt.title("SPR_BENCH Best Validation CWA2 by Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_best_CWA2_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best CWA2 bar plot: {e}")
    plt.close()

# print summary table
if summary:
    print("Best Validation CWA2 per hidden size:")
    for hd, c in summary:
        print(f"  hidden_dim={hd}: {c:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_runs = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})
batch_sizes = sorted([int(k.replace("bs", "")) for k in spr_runs.keys()])


# helper to pull arrays
def fetch(run_key, category, field):
    return [d[field] for d in spr_runs[run_key][category]]


# -------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    for bs in batch_sizes:
        rk = f"bs{bs}"
        epochs = range(1, len(spr_runs[rk]["losses"]["train"]) + 1)
        plt.plot(
            epochs,
            spr_runs[rk]["losses"]["train"],
            label=f"train bs={bs}",
            linestyle="-",
        )
        plt.plot(
            epochs,
            spr_runs[rk]["losses"]["val"],
            label=f"val   bs={bs}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss (Batch-Size Sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_batchsize.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------------------------------------------
# 2) HWA curves
try:
    plt.figure()
    for bs in batch_sizes:
        rk = f"bs{bs}"
        hwa = fetch(rk, "metrics", "hwa")
        epochs = range(1, len(hwa) + 1)
        plt.plot(epochs, hwa, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: HWA vs Epoch (Batch-Size Sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_batchsize.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# -------------------------------------------------------
# 3) Final HWA bar chart
try:
    plt.figure()
    final_hwa = [fetch(f"bs{bs}", "metrics", "hwa")[-1] for bs in batch_sizes]
    plt.bar([str(bs) for bs in batch_sizes], final_hwa)
    plt.xlabel("Batch Size")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR_BENCH: Final Harmonic Weighted Accuracy by Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

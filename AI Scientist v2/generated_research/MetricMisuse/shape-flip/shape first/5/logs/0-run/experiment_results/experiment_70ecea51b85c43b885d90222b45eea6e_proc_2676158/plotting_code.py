import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    sweep = experiment_data["BATCH_SIZE"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sweep = {}

# Gather per-batch-size aggregates
batch_sizes, test_loss, test_rcwa, test_swa, test_cwa = [], [], [], [], []
for k, v in sorted(sweep.items(), key=lambda x: int(x[0].split("_")[-1])):
    bs = int(k.split("_")[-1])
    batch_sizes.append(bs)
    test_loss.append(v.get("test_loss", np.nan))
    test_rcwa.append(v.get("test_rcwa", np.nan))
    test_swa.append(v.get("test_swa", np.nan))
    test_cwa.append(v.get("test_cwa", np.nan))

# ---------------- plotting ------------------
# 1) Test RCWA vs batch size
try:
    plt.figure()
    plt.bar(batch_sizes, test_rcwa)
    plt.xlabel("Batch Size")
    plt.ylabel("Test RCWA")
    plt.title("SPR_BENCH – Test RCWA vs Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_RCWA_vs_batchsize.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating RCWA plot: {e}")
    plt.close()

# 2) Test loss vs batch size
try:
    plt.figure()
    plt.bar(batch_sizes, test_loss, color="orange")
    plt.xlabel("Batch Size")
    plt.ylabel("Test Loss")
    plt.title("SPR_BENCH – Test Loss vs Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_loss_vs_batchsize.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) SWA & CWA vs batch size
try:
    x = np.arange(len(batch_sizes))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, test_swa, width, label="SWA")
    plt.bar(x + width / 2, test_cwa, width, label="CWA")
    plt.xticks(x, batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Score")
    plt.legend()
    plt.title("SPR_BENCH – SWA and CWA vs Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_SWA_CWA_vs_batchsize.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA/CWA plot: {e}")
    plt.close()

# 4) Train/Val loss curves for two representative batch sizes
try:
    reps = ["bs_32", "bs_256"]
    colors = {"train": "blue", "val": "red"}
    plt.figure()
    for rep in reps:
        if rep in sweep:
            ep = np.arange(1, len(sweep[rep]["losses"]["train"]) + 1)
            plt.plot(
                ep,
                sweep[rep]["losses"]["train"],
                linestyle="--",
                color=colors["train"],
                label=f"{rep}_train",
            )
            plt.plot(
                ep,
                sweep[rep]["losses"]["val"],
                linestyle="-",
                color=colors["val"],
                label=f"{rep}_val",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("SPR_BENCH – Train/Val Loss Curves (bs_32 & bs_256)")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_representative.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss-curve plot: {e}")
    plt.close()

# --------------- summary print --------------
if batch_sizes:
    idx_best = int(np.nanargmax(test_rcwa))
    print("\nBest batch size by Test RCWA:")
    print(f"  Batch size: {batch_sizes[idx_best]}")
    print(f"  Test RCWA : {test_rcwa[idx_best]:.4f}")
    print(f"  Test Loss : {test_loss[idx_best]:.4f}")
    print(f"  Test SWA  : {test_swa[idx_best]:.4f}")
    print(f"  Test CWA  : {test_cwa[idx_best]:.4f}")

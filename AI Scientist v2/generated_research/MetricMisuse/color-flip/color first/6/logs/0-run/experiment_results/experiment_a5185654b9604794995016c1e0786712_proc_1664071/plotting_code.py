import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = exp["batch_size_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# -------- figure 1: loss curves --------
try:
    fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for bs_key in sorted(runs, key=lambda x: int(x.split("_")[-1])):
        train = runs[bs_key]["losses"]["train"]  # [(epoch, loss), ...]
        val = runs[bs_key]["losses"]["val"]
        tr_x, tr_y = zip(*train)
        va_x, va_y = zip(*val)
        axs[0].plot(tr_x, tr_y, label=bs_key)
        axs[1].plot(va_x, va_y, label=bs_key)
    axs[0].set_ylabel("Train Loss")
    axs[1].set_ylabel("Val Loss")
    axs[1].set_xlabel("Epoch")
    for ax in axs:
        ax.legend()
    fig.suptitle("SPR_BENCH: Loss vs Epochs (Batch-size Tuning)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_batch_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- figure 2: validation metrics --------
try:
    metrics = ["CWA", "SWA", "EWA"]
    fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for bs_key in sorted(runs, key=lambda x: int(x.split("_")[-1])):
        vals = runs[bs_key]["metrics"]["val"]  # [(epoch, dict), ...]
        epochs = [e for e, _ in vals]
        for i, m in enumerate(metrics):
            scores = [d[m] for _, d in vals]
            axs[i].plot(epochs, scores, label=bs_key)
            axs[i].set_ylabel(m)
    axs[-1].set_xlabel("Epoch")
    for ax in axs:
        ax.legend()
    fig.suptitle("SPR_BENCH: Validation Metrics vs Epochs (Batch-size Tuning)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics_batch_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# -------- print final test metrics --------
for bs_key in sorted(runs, key=lambda x: int(x.split("_")[-1])):
    tm = runs[bs_key]["metrics"]["test"]
    print(f"{bs_key}: CWA={tm['CWA']:.3f} | SWA={tm['SWA']:.3f} | EWA={tm['EWA']:.3f}")

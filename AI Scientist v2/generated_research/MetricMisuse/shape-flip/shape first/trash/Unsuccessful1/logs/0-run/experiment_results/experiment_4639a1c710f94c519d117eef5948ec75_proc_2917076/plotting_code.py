import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data -----------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_exp = experiment_data.get("batch_size_tuning", {}).get("SPR_BENCH", {})
if not spr_exp:
    print("No SPR_BENCH data found.")
    exit()

batch_sizes = sorted([int(k.split("_")[-1]) for k in spr_exp.keys()])
tags = [f"bs_{bs}" for bs in batch_sizes]


# helper to collect per-epoch lists -----------------------------------
def get_lists(field):
    return {bs: spr_exp[f"bs_{bs}"][field] for bs in batch_sizes}


loss_tr = {bs: spr_exp[f"bs_{bs}"]["losses"]["train"] for bs in batch_sizes}
loss_val = {bs: spr_exp[f"bs_{bs}"]["losses"]["val"] for bs in batch_sizes}
hwa_val = {
    bs: [e["HWA"] for e in spr_exp[f"bs_{bs}"]["metrics"]["val"]] for bs in batch_sizes
}
test_metrics = {bs: spr_exp[f"bs_{bs}"]["metrics"]["test"] for bs in batch_sizes}

# PLOT 1: loss curves -------------------------------------------------
try:
    epochs = range(1, len(next(iter(loss_tr.values()))) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for bs in batch_sizes:
        axes[0].plot(epochs, loss_tr[bs], label=f"bs={bs}")
        axes[1].plot(epochs, loss_val[bs], label=f"bs={bs}")
    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy")
        ax.legend()
    fig.suptitle("SPR_BENCH Loss Curves\nLeft: Training, Right: Validation")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# PLOT 2: HWA validation curves --------------------------------------
try:
    epochs = range(1, len(next(iter(hwa_val.values()))) + 1)
    plt.figure(figsize=(6, 4))
    for bs in batch_sizes:
        plt.plot(epochs, hwa_val[bs], label=f"bs={bs}")
    plt.title("SPR_BENCH Validation HWA across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# PLOT 3: final test metrics -----------------------------------------
try:
    metrics = ["SWA", "CWA", "HWA"]
    x = np.arange(len(batch_sizes))
    width = 0.25
    plt.figure(figsize=(8, 4))
    for i, m in enumerate(metrics):
        vals = [test_metrics[bs][m] for bs in batch_sizes]
        plt.bar(x + (i - 1) * width, vals, width, label=m)
    plt.xticks(x, [str(bs) for bs in batch_sizes])
    plt.xlabel("Batch Size")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Test Metrics vs Batch Size")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bars: {e}")
    plt.close()

# print raw test metrics ----------------------------------------------
print("Final Test Metrics per Batch Size:")
for bs in batch_sizes:
    print(f"  bs={bs}: {test_metrics[bs]}")

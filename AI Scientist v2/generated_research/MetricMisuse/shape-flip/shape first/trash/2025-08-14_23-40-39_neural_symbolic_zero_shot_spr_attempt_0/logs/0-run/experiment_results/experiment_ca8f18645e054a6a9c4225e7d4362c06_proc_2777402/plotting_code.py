import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data ------------
exp_paths = [
    os.path.join(working_dir, "experiment_data.npy"),
    os.path.join(os.getcwd(), "experiment_data.npy"),
]
experiment_data = None
for p in exp_paths:
    if os.path.exists(p):
        try:
            experiment_data = np.load(p, allow_pickle=True).item()
            break
        except Exception as e:
            print(f"Error loading experiment data from {p}: {e}")
if experiment_data is None:
    print("experiment_data.npy not found — nothing to plot.")
    exit()

spr_runs = experiment_data.get("EMB_DIM", {}).get("SPR_BENCH", {})
if not spr_runs:
    print("No SPR_BENCH results found in experiment_data.")
    exit()

emb_sizes = sorted(spr_runs.keys())
epochs = range(1, len(next(iter(spr_runs.values()))["metrics"]["train_loss"]) + 1)

# ------------ Plot 1: train & val loss ------------
try:
    plt.figure(figsize=(10, 4))
    plt.suptitle(
        "Loss Curves by Embedding Size - SPR_BENCH\nLeft: Train, Right: Validation"
    )
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for emb in emb_sizes:
        m = spr_runs[emb]["metrics"]
        ax1.plot(epochs, m["train_loss"], label=f"emb={emb}")
        ax2.plot(epochs, m["val_loss"], label=f"emb={emb}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.legend()
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Loss")
    ax2.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------ Plot 2: validation BPS ------------
try:
    plt.figure()
    for emb in emb_sizes:
        m = spr_runs[emb]["metrics"]
        plt.plot(epochs, m["val_bps"], label=f"emb={emb}")
    plt.title("Validation BPS vs Epoch - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("BPS")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_bps.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating validation BPS plot: {e}")
    plt.close()

# ------------ Plot 3: final test BPS bar chart ------------
try:
    final_bps = [spr_runs[emb]["final_scores"]["test"]["bps"] for emb in emb_sizes]
    plt.figure()
    plt.bar([str(e) for e in emb_sizes], final_bps, color="skyblue")
    plt.title("Final Test BPS by Embedding Dimension - SPR_BENCH")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Test BPS")
    fname = os.path.join(working_dir, "SPR_BENCH_final_test_bps.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating final test BPS bar chart: {e}")
    plt.close()

# ------------ Print evaluation metric ------------
print("Final Test BPS:")
for emb, bps in zip(emb_sizes, final_bps):
    print(f"  EMB_DIM={emb}: BPS={bps:.3f}")

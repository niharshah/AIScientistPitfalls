import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment log -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

bench_key = "SPR_BENCH"
if bench_key not in experiment_data:
    print(f"{bench_key} not found in experiment_data")
    exit(0)

bench = experiment_data[bench_key]
tr_losses = bench["losses"]["train"]
val_losses = bench["losses"]["val"]
val_swa = bench["metrics"]["val"]
epoch_info = bench["epochs"]  # list of (dim, epoch)
test_swa = bench["metrics"]["test"]  # list aligned with run order

# ---------------- regroup by embedding dimension ------------------
groups = defaultdict(lambda: {"train": [], "val": [], "swa": []})
for idx, (dim, _) in enumerate(epoch_info):
    groups[dim]["train"].append(tr_losses[idx])
    groups[dim]["val"].append(val_losses[idx])
    groups[dim]["swa"].append(val_swa[idx])
dims_sorted = sorted(groups.keys())
test_swa_dict = {d: test_swa[i] for i, d in enumerate(dims_sorted)}

# ------------------ PLOT 1: loss curves ---------------------------
try:
    plt.figure()
    for d in dims_sorted:
        epochs = list(range(1, len(groups[d]["train"]) + 1))
        plt.plot(epochs, groups[d]["train"], "--", label=f"train dim={d}")
        plt.plot(epochs, groups[d]["val"], "-", label=f"val dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------ PLOT 2: validation SWA ------------------------
try:
    plt.figure()
    for d in dims_sorted:
        epochs = list(range(1, len(groups[d]["swa"]) + 1))
        plt.plot(epochs, groups[d]["swa"], marker="o", label=f"dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy (SWA)")
    plt.title("SPR_BENCH: Validation SWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_SWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve plot: {e}")
    plt.close()

# ------------------ PLOT 3: final test SWA bar --------------------
try:
    plt.figure()
    bars = [test_swa_dict[d] for d in dims_sorted]
    plt.bar([str(d) for d in dims_sorted], bars, color="steelblue")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Shape-Weighted Accuracy (SWA)")
    plt.title("SPR_BENCH: Final Test SWA per Embedding Dim")
    fname = os.path.join(working_dir, "SPR_BENCH_test_SWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test SWA bar plot: {e}")
    plt.close()

# ---------------- print summary metrics ---------------------------
print("Final Test SWA:")
for d in dims_sorted:
    print(f"dim={d}: SWA={test_swa_dict[d]:.4f}")

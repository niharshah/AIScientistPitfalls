import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely extract
def safe_get(d, *keys, default=None):
    for k in keys:
        if k not in d:
            return default
        d = d[k]
    return d


spr_dict = safe_get(experiment_data, "learning_rate", "SPR_BENCH", default={})
lrs = sorted(spr_dict.keys())  # e.g. ["5e-04","0.001","0.002","0.003"]

# Collect arrays
train_loss, val_loss, val_cwa = [], [], []
for lr in lrs:
    rec = spr_dict[lr]
    train_loss.append(rec["losses"]["train"])
    val_loss.append(rec["losses"]["val"])
    val_cwa.append(rec["metrics"]["val_cwa2"])
epochs = np.arange(1, max(len(x) for x in train_loss) + 1)

# -------------------- PLOTS --------------------
# 1. Loss curves
try:
    plt.figure()
    for tl, vl, lr in zip(train_loss, val_loss, lrs):
        plt.plot(epochs[: len(tl)], tl, label=f"train {lr}")
        plt.plot(epochs[: len(vl)], vl, "--", label=f"val {lr}")
    plt.title("SPR_BENCH: Training & Validation Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2. Validation CWA2 curves
try:
    plt.figure()
    for vc, lr in zip(val_cwa, lrs):
        plt.plot(epochs[: len(vc)], vc, label=f"{lr}")
    plt.title("SPR_BENCH: Validation CWA2 vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy (CWA2)")
    plt.legend(title="Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_valCWA2_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA2 curve plot: {e}")
    plt.close()

# 3. Final CWA2 bar chart
try:
    plt.figure()
    final_vals = [vc[-1] if len(vc) else np.nan for vc in val_cwa]
    plt.bar(range(len(lrs)), final_vals, tick_label=lrs)
    plt.title("SPR_BENCH: Final-Epoch Validation CWA2 per Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final CWA2")
    fname = os.path.join(working_dir, "SPR_BENCH_finalCWA2_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final CWA2 bar plot: {e}")
    plt.close()

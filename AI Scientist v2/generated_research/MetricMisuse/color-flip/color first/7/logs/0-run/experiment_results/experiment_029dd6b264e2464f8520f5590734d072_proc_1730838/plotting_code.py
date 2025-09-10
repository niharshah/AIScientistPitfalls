import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to fetch curves
def get_curves():
    lrs = []
    epoch_lists, loss_lists, cpx_lists, best_cpx = {}, {}, {}, {}
    for lr_str, log in experiment_data.get("learning_rate_tuning", {}).items():
        lrs.append(lr_str)
        core = log["SPR_BENCH"]
        epoch_lists[lr_str] = core["epochs"]
        loss_lists[lr_str] = core["losses"]["train"]
        cpx_lists[lr_str] = [m["cpx"] for m in core["metrics"]["val"]]
        best_cpx[lr_str] = max(cpx_lists[lr_str]) if cpx_lists[lr_str] else 0
    return lrs, epoch_lists, loss_lists, cpx_lists, best_cpx


lrs, epoch_lists, loss_lists, cpx_lists, best_cpx = get_curves()

# ------------------------------------------------------------------
# 1) Training-loss curves
try:
    plt.figure()
    for lr in lrs:
        plt.plot(epoch_lists[lr], loss_lists[lr], marker="o", label=f"lr={lr}")
    plt.title("SPR_BENCH Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_training_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating training-loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Validation CpxWA curves
try:
    plt.figure()
    for lr in lrs:
        plt.plot(epoch_lists[lr], cpx_lists[lr], marker="o", label=f"lr={lr}")
    plt.title("SPR_BENCH Validation Complexity-Weighted Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_validation_cpxwa_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Best validation CpxWA per learning rate
try:
    plt.figure()
    lr_labels = [f"lr={lr}" for lr in lrs]
    best_vals = [best_cpx[lr] for lr in lrs]
    plt.bar(lr_labels, best_vals, color="skyblue")
    plt.title("SPR_BENCH Best Validation CpxWA per Learning Rate")
    plt.ylabel("Best CpxWA")
    fname = os.path.join(working_dir, "SPR_BENCH_best_cpxwa_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating best CpxWA bar plot: {e}")
    plt.close()

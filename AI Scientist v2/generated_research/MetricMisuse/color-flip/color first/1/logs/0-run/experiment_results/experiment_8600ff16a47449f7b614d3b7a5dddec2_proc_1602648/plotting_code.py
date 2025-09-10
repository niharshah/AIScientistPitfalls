import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------ load data ------------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick exit if data missing
if not experiment_data:
    print("No experiment data found, nothing to plot.")
    exit()

spr_logs = experiment_data.get("batch_size_tuning", {}).get("SPR_BENCH", {})
if not spr_logs:
    print("SPR_BENCH logs not found in experiment_data.")
    exit()

batch_sizes = sorted(int(k.split("_")[-1]) for k in spr_logs.keys())


# ----------------------- helper -----------------------
def get_by_bs(bs):
    entry = spr_logs[f"bs_{bs}"]
    losses = entry["losses"]
    metrics = entry["metrics"]["val"]
    hmwa_curve = [m["hmwa"] for m in metrics]
    best_hmwa = max(hmwa_curve) if hmwa_curve else 0
    test_hmwa = entry.get("test_hmwa", 0)
    return losses["train"], losses["val"], hmwa_curve, best_hmwa, test_hmwa


# ----------------------- PLOT 1: Loss curves -----------------------
try:
    plt.figure()
    for bs in batch_sizes:
        tr, val, _, _, _ = get_by_bs(bs)
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"train bs={bs}")
        plt.plot(epochs, val, "--", label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train (solid)  Right: Val (dashed)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------------- PLOT 2: Dev HMWA curves -------------------
try:
    plt.figure()
    for bs in batch_sizes:
        _, _, hmwa, _, _ = get_by_bs(bs)
        epochs = range(1, len(hmwa) + 1)
        plt.plot(epochs, hmwa, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Dev HMWA")
    plt.title("SPR_BENCH Dev HMWA Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_dev_HMWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HMWA curve plot: {e}")
    plt.close()

# ----------------------- PLOT 3: Best Dev HMWA bar -----------------
try:
    best_vals = [get_by_bs(bs)[3] for bs in batch_sizes]
    plt.figure()
    plt.bar([str(bs) for bs in batch_sizes], best_vals, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Best Dev HMWA")
    plt.title("SPR_BENCH Best Dev HMWA vs Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_best_dev_HMWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best dev HMWA bar plot: {e}")
    plt.close()

# ----------------------- PLOT 4: Test HMWA bar ---------------------
try:
    test_vals = [get_by_bs(bs)[4] for bs in batch_sizes]
    plt.figure()
    plt.bar([str(bs) for bs in batch_sizes], test_vals, color="lightgreen")
    plt.xlabel("Batch Size")
    plt.ylabel("Test HMWA")
    plt.title("SPR_BENCH Test HMWA vs Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_HMWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test HMWA bar plot: {e}")
    plt.close()

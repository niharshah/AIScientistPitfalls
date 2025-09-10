import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper: safely get nested dict values
def get_nested(dic, keys, default=None):
    for k in keys:
        dic = dic.get(k, {})
    return dic or default


spr_runs = get_nested(experiment_data, ["num_epochs", "SPR_BENCH"], {})

# --------- Plot 1: Loss curves ----------
try:
    plt.figure()
    for budget, run in spr_runs.items():
        train = np.array(run["losses"]["train"])  # (epoch, loss)
        val = np.array(run["losses"]["val"])
        if train.size:
            plt.plot(train[:, 0], train[:, 1], label=f"{budget}ep-train")
        if val.size:
            plt.plot(val[:, 0], val[:, 1], "--", label=f"{budget}ep-val")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------- Plot 2: Harmonic Weighted Accuracy ----------
try:
    plt.figure()
    for budget, run in spr_runs.items():
        met = np.array(run["metrics"]["val"])  # (epoch, swa, cwa, hwa)
        if met.size:
            plt.plot(met[:, 0], met[:, 3], label=f"{budget}ep")
    plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# --------- Print best HWA for each budget ----------
for budget, run in spr_runs.items():
    met = np.array(run["metrics"]["val"])
    best_hwa = float(met[:, 3].max()) if met.size else float("nan")
    print(f"Best HWA for {budget} epochs: {best_hwa:.4f}")

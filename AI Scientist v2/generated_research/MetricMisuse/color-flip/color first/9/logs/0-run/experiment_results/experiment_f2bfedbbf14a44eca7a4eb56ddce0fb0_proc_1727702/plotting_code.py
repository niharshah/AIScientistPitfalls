import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- working dir ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("num_epochs", {}).get("SPR_BENCH", [])

# --------- figure 1 : loss curves ------------
try:
    plt.figure()
    for run in runs:
        hp = run["hyperparam_value"]
        train = np.array(run["losses"]["train"])
        val = np.array(run["losses"]["val"])
        if train.size:
            plt.plot(train[:, 0], train[:, 1], label=f"{hp}-train")
        if val.size:
            plt.plot(val[:, 0], val[:, 1], "--", label=f"{hp}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------- figure 2 : validation DWHS --------
try:
    plt.figure()
    for run in runs:
        hp = run["hyperparam_value"]
        val = np.array(run["metrics"]["val"])  # (epoch,cwa,swa,dw)
        if val.size:
            plt.plot(val[:, 0], val[:, 3], label=f"{hp}")
    plt.xlabel("Epoch")
    plt.ylabel("DWHS")
    plt.title("SPR_BENCH: Validation DWHS Across Epochs")
    plt.legend(title="num_epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_val_dwhs_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating DWHS plot: {e}")
    plt.close()

# --------- figure 3 : best DWHS summary ------
try:
    plt.figure()
    hps, best_dw = [], []
    for run in runs:
        hps.append(str(run["hyperparam_value"]))
        best_dw.append(run.get("best_val_dwhs", 0.0))
    plt.bar(hps, best_dw, color="tab:blue")
    plt.xlabel("num_epochs (hyper-param)")
    plt.ylabel("Best Validation DWHS")
    plt.title("SPR_BENCH: Best DWHS per Hyper-parameter Setting")
    fname = os.path.join(working_dir, "SPR_BENCH_best_dwhs_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()

# --------- print final metrics ---------------
for run in runs:
    hp = run["hyperparam_value"]
    be = run["best_epoch"]
    cwa, swa, dw = run.get("test_metrics", (None, None, None))
    print(f"num_epochs={hp:>2} | best_epoch={be:>2} | TEST DWHS={dw:.3f}")

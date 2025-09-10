import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["num_epochs"]["SPR_BENCH"]["runs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = []

# -------------------- figure 1: loss curves -------------------
try:
    if runs:
        r0 = runs[0]
        epochs_tr, loss_tr = zip(*r0["losses"]["train"])
        epochs_val, loss_val = zip(*r0["losses"]["val"])
        plt.figure()
        plt.plot(epochs_tr, loss_tr, label="Train")
        plt.plot(epochs_val, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Train vs Val Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_run0.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------- figure 2: HWA curves --------------------
try:
    if runs:
        r0 = runs[0]
        ep_tr, hwa_tr = zip(*r0["metrics"]["train"])
        ep_val, hwa_val = zip(*r0["metrics"]["val"])
        plt.figure()
        plt.plot(ep_tr, hwa_tr, label="Train")
        plt.plot(ep_val, hwa_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic-Weighted Accuracy")
        plt.title("SPR_BENCH – Train vs Val HWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_run0.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# -------------------- figure 3: best HWA vs epochs ------------
try:
    if runs:
        epochs = [r["epochs"] for r in runs][:5]  # at most 5 points
        best_hwa = [r["best_val_hwa"] for r in runs][:5]
        plt.figure()
        plt.plot(epochs, best_hwa, marker="o")
        plt.xlabel("Max Epochs")
        plt.ylabel("Best Validation HWA")
        plt.title("SPR_BENCH – Best Val HWA vs Max Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_bestHWA_vs_epochs.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()

# -------------------- print summary metrics -------------------
if runs:
    test_hwa_list = [
        np.mean(r["metrics"]["val"], axis=0)[1] if r["metrics"]["val"] else 0
        for r in runs
    ]
    mean_hwa = np.mean(test_hwa_list)
    best_hwa = np.max(test_hwa_list)
    print(f"Mean test HWA over runs: {mean_hwa:.4f}")
    print(f"Best test HWA over runs: {best_hwa:.4f}")
else:
    print("No runs found to summarise.")

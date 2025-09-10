import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run_key = "no_contrastive_pretraining"
ds_key = "spr"

if run_key in experiment_data and ds_key in experiment_data[run_key]:
    run = experiment_data[run_key][ds_key]
else:
    print("Requested keys not found in experiment_data.")
    run = None

# ---------- plotting ----------
if run:
    # --- 1. loss curves ---
    try:
        tr = np.array(run["losses"]["train"])  # shape (E,2)
        val = np.array(run["losses"]["val"])
        plt.figure()
        plt.plot(tr[:, 0], tr[:, 1], label="train")
        plt.plot(val[:, 0], val[:, 1], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR Classification – Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "spr_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # --- 2. metric curves (SWA, CWA, CompWA) ---
    try:
        swa = np.array(run["metrics"]["SWA"])
        cwa = np.array(run["metrics"]["CWA"])
        cpwa = np.array(run["metrics"]["CompWA"])
        plt.figure()
        plt.plot(swa[:, 0], swa[:, 1], label="SWA")
        plt.plot(cwa[:, 0], cwa[:, 1], label="CWA")
        plt.plot(cpwa[:, 0], cpwa[:, 1], label="CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR Classification – Weighted Accuracy Metrics")
        plt.legend()
        fname = os.path.join(working_dir, "spr_weighted_accuracy_metrics.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

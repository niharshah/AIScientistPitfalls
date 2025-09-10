import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load the newest experiment_data -------------
try:
    npy_files = sorted(glob.glob(os.path.join(working_dir, "experiment_data*.npy")))
    exp_path = npy_files[-1] if npy_files else None
    experiment_data = np.load(exp_path, allow_pickle=True).item() if exp_path else {}
    print(f"Loaded {exp_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = list(experiment_data.keys())
colors = plt.cm.tab10.colors


# ---------------- helper -------------
def get(dset, *keys, default=np.array([])):
    out = experiment_data[dset]
    for k in keys:
        out = out.get(k, {})
    return np.array(out) if isinstance(out, (list, tuple)) else np.array([])


# ---------------- plot 1 : loss curves per dataset -------------
for dset in datasets:
    try:
        tr = get(dset, "losses", "train")
        val = get(dset, "losses", "val")
        if tr.size == 0 or val.size == 0:
            continue
        epochs = np.arange(1, len(tr) + 1)
        plt.figure()
        plt.plot(epochs, tr, "--", color=colors[0], label="train")
        plt.plot(epochs, val, "-", color=colors[1], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset} Loss Curves\nLeft: Train (--), Right: Validation (—)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error plotting losses for {dset}: {e}")
        plt.close()

# ---------------- plot 2 : metric curves per dataset -------------
for dset in datasets:
    try:
        swa = get(dset, "metrics", "val_swa")
        cwa = get(dset, "metrics", "val_cwa")
        scaa = get(dset, "metrics", "val_scaa")
        if cwa.size == 0:
            continue
        epochs = np.arange(1, len(cwa) + 1)
        plt.figure()
        if swa.size:
            plt.plot(epochs, swa, "-", label="SWA")
        plt.plot(epochs, cwa, "-", label="CWA")
        if scaa.size:
            plt.plot(epochs, scaa, "-", label="SCAA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title(f"{dset} Validation Metrics Across Epochs\nDataset: {dset}")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_val_metrics.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error plotting metrics for {dset}: {e}")
        plt.close()

# ---------------- plot 3 : best CWA comparison -----------------
try:
    if datasets:
        best_cwas = [get(d, "metrics", "val_cwa").max() for d in datasets]
        plt.figure()
        plt.bar(
            datasets, best_cwas, color=[colors[i % 10] for i in range(len(datasets))]
        )
        plt.ylabel("Best Validation CWA")
        plt.xlabel("Dataset")
        plt.title("Best CWA Comparison Across Datasets")
        fname = os.path.join(working_dir, "best_CWA_comparison.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error plotting CWA comparison: {e}")
    plt.close()

# ---------------- print best metrics ---------------------------
for dset in datasets:
    swa_arr, cwa_arr, scaa_arr = (
        get(dset, "metrics", k) for k in ["val_swa", "val_cwa", "val_scaa"]
    )
    print(
        f"{dset}: best SWA={swa_arr.max():.3f} | best CWA={cwa_arr.max():.3f} | best SCAA={scaa_arr.max():.3f}"
    )

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- set up paths & load data --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

dataset_name = next(iter(exp.keys())) if exp else None
hist = exp.get(dataset_name, {}) if dataset_name else {}

# ------------------- print held-out metrics ----------------------
try:
    test_metrics = hist["metrics"]["test"]
    print(f"{dataset_name} test metrics:", test_metrics)
except Exception as e:
    print(f"Could not read test metrics: {e}")


# -------------- helper to reduce epoch resolution ---------------
def downsample(arr, max_pts=50):
    if len(arr) <= max_pts:
        return np.arange(1, len(arr) + 1), np.array(arr)
    idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
    return idx + 1, np.array(arr)[idx]


# --------------------------- plots -------------------------------
# 1. loss curve
try:
    tr_loss = hist["losses"]["train"]
    val_loss = hist["losses"]["val"]
    ep_tr, tr_loss_ds = downsample(tr_loss)
    ep_val, val_loss_ds = downsample(val_loss)
    plt.figure()
    plt.plot(ep_tr, tr_loss_ds, label="Train Loss")
    plt.plot(ep_val, val_loss_ds, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset_name} Loss Curve\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2-4. individual validation metrics
for metric in ["CWA", "SWA", "CompWA"]:
    try:
        vals = [m[metric] for m in hist["metrics"]["val"]]
        ep, vals_ds = downsample(vals)
        plt.figure()
        plt.plot(ep, vals_ds, label=f"Val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{dataset_name} Validation {metric}")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_name}_val_{metric}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting {metric}: {e}")
        plt.close()

# 5. combined metrics
try:
    cwa = [m["CWA"] for m in hist["metrics"]["val"]]
    swa = [m["SWA"] for m in hist["metrics"]["val"]]
    comp = [m["CompWA"] for m in hist["metrics"]["val"]]
    ep, cwa_ds = downsample(cwa)
    _, swa_ds = downsample(swa)
    _, comp_ds = downsample(comp)
    plt.figure()
    plt.plot(ep, cwa_ds, label="CWA")
    plt.plot(ep, swa_ds, label="SWA")
    plt.plot(ep, comp_ds, label="CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{dataset_name} Validation Metrics Comparison")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_val_metrics_comparison.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating combined metric plot: {e}")
    plt.close()

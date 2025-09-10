import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- pre-extract ----------
metrics_per_ds = {}
for ds_name, ds_log in experiment_data.items():
    train_loss = [v for _, v in ds_log.get("losses", {}).get("train", [])]
    val_loss = [v for _, v in ds_log.get("losses", {}).get("val", [])]
    cwa = [v for _, v in ds_log.get("metrics", {}).get("CWA", {}).get("val", [])]
    swa = [v for _, v in ds_log.get("metrics", {}).get("SWA", {}).get("val", [])]
    cpx = [v for _, v in ds_log.get("metrics", {}).get("CpxWA", {}).get("val", [])]
    metrics_per_ds[ds_name] = dict(
        train_loss=train_loss, val_loss=val_loss, CWA=cwa, SWA=swa, CpxWA=cpx
    )


# helper for plotting
def plot_metric(metric_key, ylabel, filename):
    try:
        plt.figure(figsize=(8, 5))
        for ds, m in metrics_per_ds.items():
            ep = np.arange(1, len(m[metric_key]) + 1)
            plt.plot(ep, m[metric_key], label=ds)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Validation {ylabel} Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, filename))
        plt.close()
    except Exception as e:
        print(f"Error creating plot {metric_key}: {e}")
        plt.close()


# ---------- PLOTS (≤5) ----------
# 1) train loss
try:
    plt.figure(figsize=(8, 5))
    for ds, m in metrics_per_ds.items():
        ep = np.arange(1, len(m["train_loss"]) + 1)
        plt.plot(ep, m["train_loss"], label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Across Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "allDatasets_train_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# 2) val loss
plot_metric("val_loss", "Validation Loss", "allDatasets_val_loss.png")
# 3) CWA
plot_metric("CWA", "Color Weighted Accuracy", "allDatasets_CWA.png")
# 4) SWA
plot_metric("SWA", "Shape Weighted Accuracy", "allDatasets_SWA.png")
# 5) CpxWA
plot_metric("CpxWA", "Complexity Weighted Accuracy", "allDatasets_CpxWA.png")

# ---------- print final scores ----------
for ds, m in metrics_per_ds.items():
    fl = lambda k: m[k][-1] if m[k] else float("nan")
    print(
        f"\n{ds} – Final Metrics:"
        f"  TrainLoss {fl('train_loss'):.4f}, ValLoss {fl('val_loss'):.4f},"
        f"  CWA {fl('CWA'):.4f}, SWA {fl('SWA'):.4f}, CpxWA {fl('CpxWA'):.4f}"
    )

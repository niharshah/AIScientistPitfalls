import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load experiment data ----
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# identify lr runs (keys that look like floats)
lr_keys = [k for k in exp.get("lr_sweep", {}) if k.replace(".", "", 1).isdigit()]
lr_keys.sort(key=float)

# FIGURE 1: train/val loss curves
try:
    plt.figure(figsize=(6, 4))
    for lr in lr_keys:
        tr = exp["lr_sweep"][lr]["losses"]["train"]
        vl = exp["lr_sweep"][lr]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"lr={lr} train")
        plt.plot(epochs, vl, "--", label=f"lr={lr} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Train vs Val Loss (Bag-of-Tokens Classifier)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# FIGURE 2: validation HWA curves
try:
    plt.figure(figsize=(6, 4))
    for lr in lr_keys:
        hwa = [m["HWA"] for m in exp["lr_sweep"][lr]["metrics"]["val"]]
        epochs = range(1, len(hwa) + 1)
        plt.plot(epochs, hwa, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Validation HWA across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# FIGURE 3: best test metrics bar chart
try:
    best_lr = exp["lr_sweep"].get("best_lr", None)
    if best_lr is not None:
        metrics = exp["lr_sweep"]["best_test_metrics"]
        names = list(metrics.keys())
        vals = [metrics[n] for n in names]
        plt.figure(figsize=(5, 4))
        plt.bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.title(f"SPR_BENCH: Test Metrics at Best lr={best_lr}")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(
            working_dir, f"SPR_BENCH_best_lr_{best_lr}_test_metrics.png"
        )
        plt.savefig(fname)
        plt.close()
    else:
        print("best_lr not found in experiment data.")
except Exception as e:
    print(f"Error creating best-metric bar chart: {e}")
    plt.close()

# ---- print summary ----
best_lr = exp.get("lr_sweep", {}).get("best_lr", None)
best_dev = exp.get("lr_sweep", {}).get("best_dev_hwa", None)
best_test = exp.get("lr_sweep", {}).get("best_test_metrics", None)
print(f"Best lr: {best_lr}\nBest dev HWA: {best_dev}\nBest test metrics: {best_test}")

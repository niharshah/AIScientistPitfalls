import matplotlib.pyplot as plt
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ── data loading ───────────────────────────────────────────────
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run_key = "SPR_noSeqEdge"
run = experiment_data.get(run_key, {})

loss_tr = run.get("losses", {}).get("train", [])
loss_val = run.get("losses", {}).get("val", [])
val_metrics = run.get("metrics", {}).get("val", [])
test_metrics = run.get("metrics", {}).get("test", {})

epochs = range(1, len(loss_tr) + 1)

# ── plot 1: loss curves ────────────────────────────────────────
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss – SPR_noSeqEdge")
    plt.legend()
    fname = os.path.join(working_dir, f"{run_key}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ── plot 2: validation metrics over epochs ─────────────────────
try:
    plt.figure()
    cwa = [m["CWA"] for m in val_metrics]
    swa = [m["SWA"] for m in val_metrics]
    dwa = [m["DWA"] for m in val_metrics]
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, dwa, label="DWA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Metrics – SPR_noSeqEdge")
    plt.legend()
    fname = os.path.join(working_dir, f"{run_key}_val_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()

# ── plot 3: final test metrics bar chart ───────────────────────
try:
    plt.figure()
    labels = list(test_metrics.keys())
    values = [test_metrics[k] for k in labels]
    plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Test Metrics – SPR_noSeqEdge")
    fname = os.path.join(working_dir, f"{run_key}_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# ── print test metrics for quick reference ─────────────────────
if test_metrics:
    print("Final Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.3f}")

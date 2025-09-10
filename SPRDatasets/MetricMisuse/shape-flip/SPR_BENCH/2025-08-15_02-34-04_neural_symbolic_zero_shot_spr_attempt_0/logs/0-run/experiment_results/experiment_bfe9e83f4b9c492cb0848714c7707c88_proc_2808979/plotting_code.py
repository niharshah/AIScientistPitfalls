import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    raise RuntimeError(f"Error loading experiment data: {e}")

epochs = data.get("epochs", [])
train_losses = data.get("losses", {}).get("train", [])
val_losses = data.get("losses", {}).get("val", [])
val_metrics = data.get("metrics", {}).get("val", [])
test_metrics = data.get("metrics", {}).get("test", {})


# Helper to pull metric series safely
def metric_series(metric_name):
    return [m.get(metric_name, np.nan) for m in val_metrics] if val_metrics else []


# ----------------------------- FIGURE 1 -----------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ----------------------------- FIGURE 2 -----------------------------------
try:
    hwa_vals = metric_series("hwa")
    if hwa_vals:
        plt.figure()
        plt.plot(epochs, hwa_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Validation HWA Across Epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_hwa_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating HWA curve: {e}")
    plt.close()

# ----------------------------- FIGURE 3 -----------------------------------
try:
    swa_vals, cwa_vals = metric_series("swa"), metric_series("cwa")
    if swa_vals and cwa_vals:
        plt.figure()
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("SPR_BENCH: Validation SWA and CWA Across Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_swa_cwa_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating SWA/CWA curve: {e}")
    plt.close()

# ----------------------------- FIGURE 4 -----------------------------------
try:
    if test_metrics:
        plt.figure()
        names, vals = zip(*test_metrics.items())
        plt.bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Metrics (SWA, CWA, HWA)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_test_metrics_bar.png"))
        plt.close()
except Exception as e:
    print(f"Error creating test metrics bar: {e}")
    plt.close()

# ----------------------------- PRINT METRICS ------------------------------
if test_metrics:
    print("Final Test Metrics:", test_metrics)

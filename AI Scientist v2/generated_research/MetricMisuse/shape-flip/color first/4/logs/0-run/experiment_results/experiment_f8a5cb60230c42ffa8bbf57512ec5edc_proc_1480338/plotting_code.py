import matplotlib.pyplot as plt
import numpy as np
import os

# --------------- set up & load ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR", None)
if spr is None:
    print("SPR data not found in experiment_data.npy, aborting plots.")
    exit()

losses = spr.get("losses", {})
metrics = spr.get("metrics", {})
val_metrics = metrics.get("val", [])
test_metrics = metrics.get("test", {})

# --------------- helper: epoch range ---------------
epochs = range(1, len(losses.get("train", [])) + 1)

# 1) Train vs Val loss curve ---------------------------------------
try:
    plt.figure()
    plt.plot(epochs, losses["train"], label="Train Loss")
    plt.plot(epochs, losses["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss")
    plt.legend()
    fname = "SPR_BENCH_loss_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()


# generic function to extract metric list -------------------------
def get_metric_series(metric_name):
    return [m.get(metric_name, np.nan) for m in val_metrics]


# 2) Validation CWA curve -----------------------------------------
try:
    plt.figure()
    plt.plot(epochs, get_metric_series("CWA"), label="CWA")
    plt.xlabel("Epoch")
    plt.ylabel("Color-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation CWA Over Epochs")
    fname = "SPR_BENCH_val_CWA.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating CWA curve: {e}")
    plt.close()

# 3) Validation SWA curve -----------------------------------------
try:
    plt.figure()
    plt.plot(epochs, get_metric_series("SWA"), label="SWA", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation SWA Over Epochs")
    fname = "SPR_BENCH_val_SWA.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# 4) Validation CompWA curve --------------------------------------
try:
    plt.figure()
    plt.plot(epochs, get_metric_series("CompWA"), label="CompWA", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation CompWA Over Epochs")
    fname = "SPR_BENCH_val_CompWA.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating CompWA curve: {e}")
    plt.close()

# 5) Bar chart of final test metrics ------------------------------
try:
    plt.figure()
    names = ["CWA", "SWA", "CompWA"]
    values = [test_metrics.get(k, 0.0) for k in names]
    plt.bar(names, values, color=["tab:blue", "tab:green", "tab:red"])
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Test Metrics\nLeft: CWA, Center: SWA, Right: CompWA")
    fname = "SPR_BENCH_test_metrics_bar.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()

# --------------- print final test metrics ------------------------
print("SPR_BENCH final test metrics:", test_metrics)

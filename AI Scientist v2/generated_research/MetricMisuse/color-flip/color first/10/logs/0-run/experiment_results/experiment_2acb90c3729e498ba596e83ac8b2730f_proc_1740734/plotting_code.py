import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    bench = experiment_data.get("SPR_BENCH", {})
    metrics = bench.get("metrics", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    bench, metrics = {}, {}


# Helper to safely fetch a list
def get(key):
    return metrics.get(key, [])


epochs = range(1, len(get("train_loss")) + 1)

# ---------- plots ----------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, get("train_loss"), label="train_loss")
    plt.plot(epochs, get("val_loss"), label="val_loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Weighted accuracy curves
try:
    plt.figure()
    plt.plot(epochs, get("val_CWA"), label="CWA")
    plt.plot(epochs, get("val_SWA"), label="SWA")
    plt.plot(epochs, get("val_CWA2"), label="CWA2")
    plt.title("SPR_BENCH Validation Weighted Accuracies\nRight: CWA/SWA/CWA2 Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_weighted_acc_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 3) Final epoch bar chart
try:
    plt.figure()
    labels = ["CWA", "SWA", "CWA2"]
    finals = [
        get("val_CWA")[-1] if get("val_CWA") else 0,
        get("val_SWA")[-1] if get("val_SWA") else 0,
        get("val_CWA2")[-1] if get("val_CWA2") else 0,
    ]
    plt.bar(labels, finals)
    plt.title("SPR_BENCH Final Epoch Weighted Accuracies")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_final_weighted_acc.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final bar plot: {e}")
    plt.close()

# ---------- print summary ----------
if finals:
    print(
        f"Final metrics - CWA: {finals[0]:.4f}, SWA: {finals[1]:.4f}, CWA2: {finals[2]:.4f}"
    )
else:
    print("No metrics found.")

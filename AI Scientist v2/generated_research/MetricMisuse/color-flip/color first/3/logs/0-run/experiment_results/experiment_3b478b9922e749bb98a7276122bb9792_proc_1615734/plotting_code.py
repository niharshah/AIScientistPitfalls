import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load data ---------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Navigate to the block we need
model_key, dataset_key = "MeanPoolEncoder", "SPR_BENCH"
data = experiment_data.get(model_key, {}).get(dataset_key, {})

loss_train = data.get("losses", {}).get("train", [])  # list of (epoch, loss)
loss_val = data.get("losses", {}).get("val", [])  # list of (epoch, loss)
metrics_val = data.get("metrics", {}).get(
    "val", []
)  # list of (epoch,CWA,SWA,HCSA,SNWA)


# Helper to convert list-of-tuples -> np arrays (may be empty)
def to_xy(arr, idx=1):
    if not arr:
        return np.array([]), np.array([])
    arr = np.array(arr)
    return arr[:, 0], arr[:, idx]


# ---------------- Plot 1: Loss curves ---------------- #
try:
    ep_tr, loss_tr = to_xy(loss_train)
    ep_val, loss_v = to_xy(loss_val)
    if ep_tr.size and ep_val.size:
        plt.figure()
        plt.plot(ep_tr, loss_tr, label="Train")
        plt.plot(ep_val, loss_v, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Loss data missing, skipping loss plot.")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------- Plot 2: Metric curves ---------------- #
try:
    ep, hcs = to_xy(metrics_val, 3)  # HCSA index 3
    _, snw = to_xy(metrics_val, 4)  # SNWA  index 4
    if ep.size:
        plt.figure()
        plt.plot(ep, hcs, label="HCSA")
        plt.plot(ep, snw, label="SNWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation Metrics (HCSA & SNWA)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_metrics_curves.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Metric data missing, skipping metric plot.")
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------------- Print final metrics ---------------- #
if metrics_val:
    last_epoch, _, _, last_hcs, last_snw = metrics_val[-1]
    print(
        f"Final validation epoch {last_epoch} -> HCSA={last_hcs:.3f}, SNWA={last_snw:.3f}"
    )

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
exp_file = os.path.join(working_dir, "experiment_data.npy")

# ---------- load ----------
try:
    exp = np.load(exp_file, allow_pickle=True).item()
    ed = exp["NodeFeatureProjectionAblation"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = ed["epochs"]
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
ds_name = "SPR_BENCH"


# Helper for plotting
def _plot(y_tr, y_val, ylabel, fname_suffix):
    try:
        plt.figure()
        plt.plot(epochs, y_tr, label="train")
        plt.plot(epochs, y_val, label="val")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ds_name} – {ylabel} (Train vs Validation)")
        plt.legend()
        save_path = os.path.join(working_dir, f"{ds_name}_{fname_suffix}_{ts}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {ylabel}: {e}")
        plt.close()


# ---------- plots ----------
_plot(ed["losses"]["train"], ed["losses"]["val"], "Loss", "loss_curve")
_plot(
    ed["metrics"]["train"]["CWA"],
    ed["metrics"]["val"]["CWA"],
    "Color-Weighted Accuracy",
    "cwa_curve",
)
_plot(
    ed["metrics"]["train"]["SWA"],
    ed["metrics"]["val"]["SWA"],
    "Shape-Weighted Accuracy",
    "swa_curve",
)
_plot(
    ed["metrics"]["train"]["CplxWA"],
    ed["metrics"]["val"]["CplxWA"],
    "Complexity-Weighted Accuracy",
    "cplxwa_curve",
)

# ---------- print test metrics ----------
t = ed["metrics"]["test"]
print(f"Test CWA={t['CWA']:.3f}, SWA={t['SWA']:.3f}, CplxWA={t['CplxWA']:.3f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------
try:
    spr_exp = experiment_data["SPR"]
except Exception as e:
    print(f"SPR data not found: {e}")
    spr_exp = {}


# -------------- helper to save & close in one place ------------
def _save_close(fig_name):
    plt.savefig(os.path.join(working_dir, fig_name))
    plt.close()


# -------------------- 1. Loss Curves ---------------------------
try:
    tr_loss = spr_exp["losses"]["train"]
    val_loss = spr_exp["losses"]["val"]
    epochs = range(1, len(tr_loss) + 1)

    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR: Training vs Validation Loss")
    plt.legend()
    _save_close("SPR_loss_curves.png")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------- 2. Val Metric Curves ---------------------
try:
    val_metrics = spr_exp["metrics"]["val"]  # list of dicts per epoch
    cwa = [m["CWA"] for m in val_metrics]
    swa = [m["SWA"] for m in val_metrics]
    comp = [m["CompWA"] for m in val_metrics]
    epochs = range(1, len(cwa) + 1)

    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, comp, label="CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR: Validation Weighted Accuracies")
    plt.legend()
    _save_close("SPR_val_weighted_accuracies.png")
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# -------------------- 3. Test Metric Bars ----------------------
try:
    test_m = spr_exp["metrics"]["test"]  # dict with 3 metrics
    labels = list(test_m.keys())
    values = [test_m[k] for k in labels]

    plt.figure()
    plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylim(0, 1)
    plt.title("SPR: Final Test Weighted Accuracies")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    _save_close("SPR_test_weighted_accuracies.png")
except Exception as e:
    print(f"Error creating test bar plot: {e}")
    plt.close()

# -------------------- 4. Overlay Loss (zoom) -------------------
try:
    tr_loss = spr_exp["losses"]["train"]
    val_loss = spr_exp["losses"]["val"]
    epochs = range(1, len(tr_loss) + 1)

    plt.figure()
    plt.plot(epochs, tr_loss, "--", label="Train Loss")
    plt.plot(epochs, val_loss, "-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR: Overlaid Loss Curves (zoom view)")
    plt.legend()
    plt.ylim(min(val_loss) * 0.9, max(tr_loss) * 1.1)
    _save_close("SPR_loss_overlay_zoom.png")
except Exception as e:
    print(f"Error creating overlay loss plot: {e}")
    plt.close()

# -------------------- 5. Print Final Metrics -------------------
try:
    print("Final SPR Test Metrics:", spr_exp["metrics"]["test"])
except Exception as e:
    print(f"Error printing final metrics: {e}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch nested dict safely
def get(dic, *ks, default=None):
    for k in ks:
        dic = dic.get(k, {})
    return dic if dic else default


ds_key = "SPR_BENCH"
mdl_key = "max_pool"
ed = get(experiment_data, mdl_key, ds_key, default={})

# ---------- plots ----------
# 1. Loss curves
try:
    epochs = list(range(1, len(ed["losses"]["train"]) + 1))
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"], label="Train")
    plt.plot(epochs, ed["losses"]["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"{ds_key}: Loss Curves (Transformer MaxPool)")
    plt.legend()
    plt.tight_layout()
    fname = f"{ds_key}_loss_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2. MCC curves
try:
    epochs = list(range(1, len(ed["metrics"]["train_MCC"]) + 1))
    plt.figure()
    plt.plot(epochs, ed["metrics"]["train_MCC"], label="Train MCC")
    plt.plot(epochs, ed["metrics"]["val_MCC"], label="Val MCC")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews CorrCoef")
    plt.title(f"{ds_key}: MCC Curves (Transformer MaxPool)")
    plt.legend()
    plt.tight_layout()
    fname = f"{ds_key}_MCC_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating MCC plot: {e}")
    plt.close()

# 3. Test metric comparison
try:
    plt.figure()
    metrics = ["Test_MCC", "Test_F1"]
    values = [ed.get("test_MCC", 0), ed.get("test_F1", 0)]
    plt.bar(metrics, values, color=["steelblue", "orange"])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.title(f"{ds_key}: Test Metrics (Transformer MaxPool)")
    plt.tight_layout()
    fname = f"{ds_key}_test_metrics.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# ---------- print evaluation ----------
if ed:
    print(
        f"Test MCC: {ed.get('test_MCC', 'NA'):.3f}, Test Macro-F1: {ed.get('test_F1', 'NA'):.3f}"
    )

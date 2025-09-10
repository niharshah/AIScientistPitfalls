import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- basic setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data -----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["collapsed_edge_type"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data:  # proceed only if data is available
    epochs = range(1, len(spr_data["losses"]["train"]) + 1)

    # -------------- loss curves ----------------
    try:
        plt.figure()
        plt.plot(epochs, spr_data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, spr_data["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Loss Curves\nTrain vs Validation")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------- validation metrics curves -------
    try:
        plt.figure()
        vals = spr_data["metrics"]["val"]
        cwa = [m["CWA"] for m in vals]
        swa = [m["SWA"] for m in vals]
        comp = [m["CompWA"] for m in vals]
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, comp, label="CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR Validation Metrics over Epochs")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_val_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # -------------- test metrics --------------
    try:
        plt.figure()
        test_m = spr_data["metrics"]["test"]
        names = list(test_m.keys())
        vals = [test_m[k] for k in names]
        plt.bar(names, vals, color=["steelblue", "salmon", "seagreen"])
        plt.ylim(0, 1)
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR Test Metrics Summary")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # --------------- print metrics ------------
    print("Final SPR test metrics:", spr_data["metrics"]["test"])

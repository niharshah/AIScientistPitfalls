import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("No-SHC", {}).get("SPR", {})
losses = spr.get("losses", {})
metrics = spr.get("metrics", {})

# ------------------ loss curves -----------------
try:
    tr = np.array(losses.get("train", []))
    va = np.array(losses.get("val", []))
    if tr.size and va.size:
        plt.figure()
        plt.plot(tr[:, 0], tr[:, 1], label="Train")
        plt.plot(va[:, 0], va[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR Dataset - Loss Curves (No-SHC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_No-SHC_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------- metric curves -------------------
try:
    mv = np.array(metrics.get("val", []))
    if mv.size:
        epochs, cwa, swa, hm, ocga = mv.T
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, hm, label="HMean")
        plt.plot(epochs, ocga, label="OCGA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Dataset - Validation Metrics (No-SHC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_No-SHC_metric_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

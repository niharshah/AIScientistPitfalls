import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = {}


# quick helpers
def arr(path):
    cur = spr
    for p in path:
        cur = cur[p]
    return np.array(cur)


# --------------- PLOT 1: losses ---------------
try:
    tr_loss = arr(["losses", "train"])
    val_loss = arr(["losses", "val"])
    epochs = np.arange(1, len(tr_loss) + 1)
    plt.figure()
    plt.plot(epochs, tr_loss, "r--", label="Train")
    plt.plot(epochs, val_loss, "r-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Loss Curves\nLeft: Train (--), Right: Validation (—)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"), dpi=200)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------------- PLOT 2: SWA vs CWA ----------
try:
    swa = arr(["metrics", "train"])
    cwa = arr(["metrics", "val"])
    epochs = np.arange(1, len(swa) + 1)
    plt.figure()
    plt.plot(epochs, swa, "b--", label="SWA (train)")
    plt.plot(epochs, cwa, "b-", label="CWA (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR Weighted Accuracy Curves\nLeft: SWA (--), Right: CWA (—)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_weighted_accuracy_curves.png"), dpi=200)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# --------------- PLOT 3: SCAA ----------------
try:
    scaa = arr(["SCAA", "val"])
    epochs = np.arange(1, len(scaa) + 1)
    plt.figure()
    plt.plot(epochs, scaa, "g-")
    plt.xlabel("Epoch")
    plt.ylabel("SCAA")
    plt.title("SPR Validation SCAA Across Epochs\nDataset: SPR")
    plt.savefig(os.path.join(working_dir, "SPR_SCAA_curves.png"), dpi=200)
    plt.close()
except Exception as e:
    print(f"Error creating SCAA plot: {e}")
    plt.close()

# ------------- print best SCAA ----------------
if scaa.size:
    best_ep = scaa.argmax() + 1
    print(f"Best Val SCAA: {scaa.max():.3f} at epoch {best_ep}")

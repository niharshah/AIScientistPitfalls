import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

root = experiment_data.get("bow_shuffle", {}).get("SPR_BENCH", {})

# -------- plot 1: loss curves --------
try:
    plt.figure(figsize=(7, 5))
    for hs, res in root.items():
        if not isinstance(hs, int):
            continue
        epochs_tr, losses_tr = zip(*res["losses"]["train"])
        epochs_v, losses_v = zip(*res["losses"]["val"])
        plt.plot(epochs_tr, losses_tr, label=f"train_h{hs}")
        plt.plot(epochs_v, losses_v, "--", label=f"val_h{hs}")
    plt.title("SPR_BENCH Loss Curves (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- plot 2: weighted accuracy metrics --------
try:
    plt.figure(figsize=(7, 5))
    for hs, res in root.items():
        if not isinstance(hs, int):
            continue
        epochs, swa, cwa, hwa = zip(*res["metrics"]["val"])
        plt.plot(epochs, hwa, label=f"HWA_h{hs}")
    plt.title("SPR_BENCH Harmonic Weighted Accuracy (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

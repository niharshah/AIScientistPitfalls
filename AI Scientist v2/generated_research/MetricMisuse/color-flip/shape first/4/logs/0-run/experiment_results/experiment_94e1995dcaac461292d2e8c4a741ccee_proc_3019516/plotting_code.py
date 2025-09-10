import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

lr_dict = experiment_data.get("learning_rate", {})

# -------- plot losses --------
try:
    plt.figure()
    for lr_tag, rec in lr_dict.items():
        bench = rec["SPR_BENCH"]
        tr = bench["losses"]["train"]
        val = bench["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{lr_tag} train")
        plt.plot(epochs, val, "--", label=f"{lr_tag} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nSolid: Train, Dashed: Validation")
    plt.legend(fontsize="small")
    f_name = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(f_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {f_name}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------- plot HWA metric --------
try:
    plt.figure()
    for lr_tag, rec in lr_dict.items():
        bench = rec["SPR_BENCH"]
        hwa = bench["metrics"]["val"]  # list of HWA per epoch
        epochs = range(1, len(hwa) + 1)
        plt.plot(epochs, hwa, label=f"{lr_tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH Validation HWA vs Epoch\nOne line per Learning Rate")
    plt.legend(title="Learning rate", fontsize="small")
    f_name = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
    plt.savefig(f_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {f_name}")
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

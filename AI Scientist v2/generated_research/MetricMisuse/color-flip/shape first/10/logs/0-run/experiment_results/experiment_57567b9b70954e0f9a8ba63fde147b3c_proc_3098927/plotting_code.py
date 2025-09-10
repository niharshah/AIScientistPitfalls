import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------- load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dr_dict = experiment_data.get("dropout_rate", {})

# ------------------------------------------------- PLOT 1: train/val loss
try:
    plt.figure()
    for dr, rec in dr_dict.items():
        epochs = np.arange(1, len(rec["metrics"]["train_loss"]) + 1)
        plt.plot(epochs, rec["metrics"]["train_loss"], label=f"train dr={dr}")
        plt.plot(epochs, rec["metrics"]["val_loss"], "--", label=f"val   dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_Synth – Loss Curves\nTrain vs Val for different dropout rates")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_Synth_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ------------------------------------------------- PLOT 2: HWA over epochs
try:
    plt.figure()
    for dr, rec in dr_dict.items():
        epochs = np.arange(1, len(rec["metrics"]["HWA"]) + 1)
        plt.plot(epochs, rec["metrics"]["HWA"], label=f"HWA dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_Synth – Harmonic Weighted Accuracy over Training")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_Synth_HWA_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves plot: {e}")
    plt.close()

# ------------------------------------------------- PLOT 3: Final HWA vs dropout
try:
    plt.figure()
    dr_values, final_hwa = [], []
    for dr, rec in sorted(dr_dict.items(), key=lambda x: float(x[0])):
        dr_values.append(float(dr))
        final_hwa.append(rec["metrics"]["HWA"][-1])
    plt.bar(dr_values, final_hwa, width=0.05, color="tab:blue")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_Synth – Final HWA by Dropout Rate")
    fname = os.path.join(working_dir, "SPR_Synth_final_HWA_vs_dropout.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

print("All plots saved to", working_dir)

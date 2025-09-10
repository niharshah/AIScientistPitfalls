import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------  Load data  ---------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("dropout_rate", {}).get("SPR_BENCH", {})
dropouts = sorted(runs.keys())

# ------------------- Figure 1: Loss curves ---------------------------------
try:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for dr in dropouts:
        epochs = range(1, len(runs[dr]["losses"]["train"]) + 1)
        ax[0].plot(epochs, runs[dr]["losses"]["train"], label=f"dr={dr}")
        ax[1].plot(epochs, runs[dr]["losses"]["val"], label=f"dr={dr}")
    ax[0].set_title("Left: Training Loss")
    ax[1].set_title("Right: Validation Loss")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Loss")
        a.legend(fontsize=6)
    fig.suptitle("Loss Curves across Dropout Rates (SPR_BENCH)")
    fig.tight_layout()
    fig.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# ------------------- Figure 2: Validation HWA per epoch --------------------
try:
    plt.figure(figsize=(6, 4))
    for dr in dropouts:
        epochs = range(1, len(runs[dr]["metrics"]["val"]) + 1)
        plt.plot(epochs, runs[dr]["metrics"]["val"], label=f"dr={dr}")
    plt.title("Validation HWA vs Epoch (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_HWA_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA-per-epoch figure: {e}")
    plt.close()

# ------------- Figure 3: Final HWA vs dropout ------------------------------
try:
    final_hwa = [runs[dr]["metrics"]["val"][-1] for dr in dropouts]
    plt.figure(figsize=(6, 4))
    plt.plot(dropouts, final_hwa, marker="o")
    plt.title("Final-Epoch Validation HWA vs Dropout Rate (SPR_BENCH)")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_HWA_vs_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final HWA figure: {e}")
    plt.close()

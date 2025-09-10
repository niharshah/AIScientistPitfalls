import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    bs_dict = experiment_data["batch_size"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    bs_dict = {}

# ---------- Figure 1: Loss curves ----------
try:
    plt.figure(figsize=(8, 5))
    for exp_key, data in bs_dict.items():
        epochs_tr, loss_tr = zip(*data["losses"]["train"])
        epochs_va, loss_va = zip(*data["losses"]["val"])
        bs = exp_key.split("bs")[-1]
        plt.plot(epochs_tr, loss_tr, label=f"Train bs{bs}", linestyle="-")
        plt.plot(epochs_va, loss_va, label=f"Val bs{bs}", linestyle="--")
    plt.title("Synthetic SPR_BENCH – Training/Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Figure 2: CSHM curves ----------
try:
    plt.figure(figsize=(8, 5))
    for exp_key, data in bs_dict.items():
        epochs, cwa, swa, cshm = zip(*data["metrics"]["val"])
        bs = exp_key.split("bs")[-1]
        plt.plot(epochs, cshm, label=f"CSHM bs{bs}")
    plt.title("Synthetic SPR_BENCH – CSHM (Harmonic Mean) vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("CSHM")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_cshm_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating CSHM plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper to fetch SPR_BENCH results
bench_key = "SPR_BENCH"
sweep = experiment_data.get("max_grad_norm", {}).get(bench_key, {})

# Figure 1: Loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title("Left: Training Loss")
    axes[1].set_title("Right: Validation Loss")
    for clip_key, log in sweep.items():
        epochs = np.arange(1, len(log["losses"]["train"]) + 1)
        axes[0].plot(epochs, log["losses"]["train"], label=f"clip={clip_key}")
        axes[1].plot(epochs, log["losses"]["val"], label=f"clip={clip_key}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("SPR_BENCH Loss Curves")
    fig.tight_layout()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_path)
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Figure 2: Validation CWA curves
try:
    plt.figure(figsize=(6, 4))
    for clip_key, log in sweep.items():
        epochs = np.arange(1, len(log["metrics"]["val"]) + 1)
        plt.plot(epochs, log["metrics"]["val"], label=f"clip={clip_key}")
    plt.title("SPR_BENCH Validation Complexity-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(working_dir, "SPR_BENCH_val_CWA_curves.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating CWA curves: {e}")
    plt.close()

# Print final CWA per clipping value
for clip_key, log in sweep.items():
    final_cwa = log["metrics"]["val"][-1] if log["metrics"]["val"] else float("nan")
    print(f"Final CWA for clip={clip_key}: {final_cwa:.4f}")

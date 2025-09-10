import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = list(experiment_data["mean_pool"]["SPR"]["num_epochs"].items())[
        :5
    ]  # limit to 5
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = []

# ---------- Plot 1: Loss curves ----------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, data in runs:
        axes[0].plot(data["losses"]["train"], label=name)
        axes[1].plot(data["losses"]["val"], label=name)
    for ax, sub in zip(axes, ["Training", "Validation"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(sub)
    fig.suptitle("SPR GRU (mean-pool) Loss Curves\nLeft: Training, Right: Validation")
    axes[0].legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(working_dir, "SPR_mean_pool_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- Plot 2: HWA curves ----------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, data in runs:
        hwa_tr = [m[2] for m in data["metrics"]["train"]]
        hwa_val = [m[2] for m in data["metrics"]["val"]]
        axes[0].plot(hwa_tr, label=name)
        axes[1].plot(hwa_val, label=name)
    for ax, sub in zip(axes, ["Training", "Validation"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("HWA")
        ax.set_title(sub)
    fig.suptitle("SPR GRU (mean-pool) HWA Curves\nLeft: Training, Right: Validation")
    axes[0].legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(working_dir, "SPR_mean_pool_hwa_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# ---------- Plot 3: Test HWA bar chart ----------
try:
    labels = [name for name, _ in runs]
    test_hwa = [data["metrics"]["test"][2] for _, data in runs]
    fig = plt.figure(figsize=(6, 4))
    plt.bar(labels, test_hwa, color="skyblue")
    plt.ylabel("Test HWA")
    plt.xlabel("Run (num_epochs)")
    plt.title("SPR GRU (mean-pool) Test HWA per Configuration")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_mean_pool_test_hwa.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close(fig)
except Exception as e:
    print(f"Error creating Test HWA bar chart: {e}")
    plt.close()

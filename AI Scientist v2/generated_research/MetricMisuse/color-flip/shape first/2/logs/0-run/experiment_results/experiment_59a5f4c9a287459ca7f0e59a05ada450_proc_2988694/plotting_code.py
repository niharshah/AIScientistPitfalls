import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

lr_dict = experiment_data.get("learning_rate", {})
tags = sorted(lr_dict.keys())  # e.g. ['lr_0.0003', 'lr_0.0005', ...]


# Helper to grab per-epoch arrays
def get_list(tag, key_path):
    d = lr_dict[tag]
    obj = d
    for k in key_path:
        obj = obj[k]
    return obj


# 1) Train / Val loss curves
try:
    plt.figure()
    for tag in tags:
        train_losses = get_list(tag, ["losses", "train"])
        val_losses = get_list(tag, ["losses", "val"])
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f"{tag}-train")
        plt.plot(epochs, val_losses, label=f"{tag}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Learning Rate Sweep – Train vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "lr_sweep_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) Validation HWA curves
try:
    plt.figure()
    for tag in tags:
        hwas = [m["hwa"] for m in get_list(tag, ["metrics", "val"])]
        epochs = range(1, len(hwas) + 1)
        plt.plot(epochs, hwas, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy (HWA)")
    plt.title("Learning Rate Sweep – Validation HWA per Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "lr_sweep_val_hwa_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# 3) Final-epoch HWA comparison
try:
    plt.figure()
    final_hwas = [get_list(tag, ["metrics", "val"])[-1]["hwa"] for tag in tags]
    plt.bar(tags, final_hwas, color="skyblue")
    plt.ylabel("Final HWA")
    plt.title("Learning Rate Sweep – Final Epoch HWA")
    plt.xticks(rotation=45)
    fname = os.path.join(working_dir, "lr_sweep_final_hwa_bar.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# Print final HWA values
for tag, hwa in zip(tags, final_hwas):
    print(f"{tag}: final HWA = {hwa:.4f}")

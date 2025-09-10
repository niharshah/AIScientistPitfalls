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

# Extract useful info
batch_sizes = sorted([int(k.split("_")[-1]) for k in experiment_data.keys()])
train_losses, val_losses, val_hwas = {}, {}, {}
for bs in batch_sizes:
    key = f"batch_size_{bs}"
    train_losses[bs] = experiment_data[key]["losses"]["train"]
    val_losses[bs] = experiment_data[key]["losses"]["val"]
    val_hwas[bs] = experiment_data[key]["metrics"]["val"]

# 1. Loss curves
try:
    plt.figure()
    for bs in batch_sizes:
        epochs = range(1, len(train_losses[bs]) + 1)
        plt.plot(epochs, train_losses[bs], "--", label=f"Train bs={bs}")
        plt.plot(epochs, val_losses[bs], "-", label=f"Val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Dataset – Training vs. Validation Loss\n(different batch sizes)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2. Validation HWA curves
try:
    plt.figure()
    for bs in batch_sizes:
        epochs = range(1, len(val_hwas[bs]) + 1)
        plt.plot(epochs, val_hwas[bs], label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR Dataset – Validation HWA Across Epochs\n(different batch sizes)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_hwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves plot: {e}")
    plt.close()

# 3. Final HWA bar chart
try:
    plt.figure()
    final_hwas = [val_hwas[bs][-1] if val_hwas.get(bs) else 0 for bs in batch_sizes]
    plt.bar([str(bs) for bs in batch_sizes], final_hwas, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR Dataset – Final Validation HWA per Batch Size")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_final_hwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

print("Plots saved to:", working_dir)

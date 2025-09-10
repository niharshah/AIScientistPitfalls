import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_runs = experiment_data["batch_size"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_runs = {}

# gather metrics
batch_sizes = sorted(spr_runs.keys())
train_losses, val_losses, val_hwas, test_hwa = {}, {}, {}, {}
for bs in batch_sizes:
    run = spr_runs[bs]
    train_losses[bs] = run["losses"]["train"]
    val_losses[bs] = run["losses"]["val"]
    val_hwas[bs] = run["metrics"]["val"]
    test_hwa[bs] = run["test_metrics"]

# 1. Loss curves
try:
    plt.figure(figsize=(6, 4))
    for bs in batch_sizes:
        epochs = np.arange(1, len(train_losses[bs]) + 1)
        plt.plot(epochs, train_losses[bs], label=f"train bs={bs}")
        plt.plot(epochs, val_losses[bs], linestyle="--", label=f"val bs={bs}")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=6)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_loss_curves_all_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2. Validation HWA curves
try:
    plt.figure(figsize=(6, 4))
    for bs in batch_sizes:
        epochs = np.arange(1, len(val_hwas[bs]) + 1)
        plt.plot(epochs, val_hwas[bs], label=f"bs={bs}")
    plt.title("SPR_BENCH Validation HWA over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=6)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_val_hwa_curves_all_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# 3. Test HWA vs batch size
try:
    plt.figure(figsize=(5, 3))
    bars = [test_hwa[bs] for bs in batch_sizes]
    plt.bar(range(len(batch_sizes)), bars, tick_label=batch_sizes)
    plt.title("SPR_BENCH Test HWA vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("HWA")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_test_hwa_vs_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test HWA bar plot: {e}")
    plt.close()

# print evaluation summary
print("Final Test HWA by Batch Size:", test_hwa)

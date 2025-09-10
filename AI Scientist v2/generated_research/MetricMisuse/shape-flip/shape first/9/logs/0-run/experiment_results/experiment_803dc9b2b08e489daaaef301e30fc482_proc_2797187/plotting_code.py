import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- Load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_tuning = experiment_data.get("lr_tuning", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_tuning = {}

# Helper: gather per-LR arrays
lrs, train_losses, val_losses, val_hwa, test_hwa = [], [], [], [], []
for lr_key, blob in lr_tuning.items():
    lrs.append(lr_key)
    train_losses.append(blob["losses"]["train"])
    val_losses.append(blob["losses"]["val"])
    val_hwa.append(blob["metrics"]["val"])
    test_hwa.append(blob["metrics"].get("test", np.nan))

# -------------------- Plot 1: Loss curves --------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lr_key, tr, vl in zip(lrs, train_losses, val_losses):
        epochs = np.arange(1, len(tr) + 1)
        axes[0].plot(epochs, tr, label=lr_key)
        axes[1].plot(epochs, vl, label=lr_key)
    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
    fig.suptitle("SPR Loss Curves – Left: Train, Right: Validation")
    fig.legend(loc="upper center", ncol=len(lrs))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------------------- Plot 2: Validation HWA curves --------------------
try:
    plt.figure(figsize=(6, 4))
    for lr_key, hwa in zip(lrs, val_hwa):
        plt.plot(np.arange(1, len(hwa) + 1), hwa, label=lr_key)
    plt.title("SPR Validation HWA Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_val_HWA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# -------------------- Plot 3: Test HWA bar chart --------------------
try:
    plt.figure(figsize=(6, 4))
    x = np.arange(len(lrs))
    plt.bar(x, test_hwa, color="skyblue")
    plt.xticks(x, lrs, rotation=45)
    plt.ylabel("HWA")
    plt.title("SPR Test HWA per Learning Rate")
    for i, v in enumerate(test_hwa):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_test_HWA_bars.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test HWA bar chart: {e}")
    plt.close()

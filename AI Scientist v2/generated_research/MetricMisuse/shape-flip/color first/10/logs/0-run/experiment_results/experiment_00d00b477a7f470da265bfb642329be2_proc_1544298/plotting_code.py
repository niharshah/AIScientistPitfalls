import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
lrs = data.get("lr_values", [])
train_losses = data.get("losses", {}).get("train", [])
val_losses = data.get("losses", {}).get("val", [])
val_metrics = data.get("metrics", {}).get("val", [])

# helper: extract HWA curves
hwa_curves = []
for mlist in val_metrics:
    hwa_curves.append([ep_dict["hwa"] for ep_dict in mlist])

# ---------- figure 1: loss curves ----------
try:
    plt.figure(figsize=(6, 4))
    for lr, tr, va in zip(lrs, train_losses, val_losses):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"lr={lr} train", linewidth=1.5)
        plt.plot(epochs, va, linestyle="--", label=f"lr={lr} val", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nSolid: Train, Dashed: Validation")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- figure 2: HWA curves ----------
try:
    plt.figure(figsize=(6, 4))
    for lr, hwa in zip(lrs, hwa_curves):
        epochs = range(1, len(hwa) + 1)
        plt.plot(epochs, hwa, label=f"lr={lr}", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.title("SPR_BENCH HWA over Epochs\nLines: Validation HWA per LR")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# ---------- figure 3: final HWA bar chart ----------
try:
    final_hwa = [curve[-1] if curve else 0 for curve in hwa_curves]
    plt.figure(figsize=(6, 4))
    plt.bar([str(lr) for lr in lrs], final_hwa, color="skyblue")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Dev HWA")
    plt.title("SPR_BENCH Final Dev HWA vs Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# ---------- print numerical summary ----------
for lr, hwa in zip(lrs, final_hwa):
    print(f"LR {lr:>6}: Final Dev HWA = {hwa:.3f}")

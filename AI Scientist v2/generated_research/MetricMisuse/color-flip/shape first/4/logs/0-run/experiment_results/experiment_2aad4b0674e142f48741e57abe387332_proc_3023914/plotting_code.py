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

spr_histories = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
if not spr_histories:
    print("No SPR_BENCH data found, nothing to plot.")
    exit()

lrs = sorted(spr_histories.keys(), key=float)
epochs = len(next(iter(spr_histories.values()))["losses"]["train"])

# ---------- collect metrics ----------
train_losses, val_losses, val_hwas = {}, {}, {}
for lr in lrs:
    h = spr_histories[lr]
    train_losses[lr] = h["losses"]["train"]
    val_losses[lr] = h["losses"]["val"]
    val_hwas[lr] = h["metrics"]["val"]

# ---------- Plot 1: Loss curves ----------
try:
    plt.figure(figsize=(8, 5))
    for lr in lrs:
        ep = np.arange(1, len(train_losses[lr]) + 1)
        plt.plot(ep, train_losses[lr], label=f"train lr={lr}")
        plt.plot(ep, val_losses[lr], "--", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation (all LRs)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Plot 2: Validation HWA ----------
try:
    plt.figure(figsize=(8, 5))
    for lr in lrs:
        ep = np.arange(1, len(val_hwas[lr]) + 1)
        plt.plot(ep, val_hwas[lr], label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title(
        "SPR_BENCH Validation HWA Across Epochs\nLeft: Ground Truth metric, Right: Generated metric curves"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- Plot 3: Final HWA bar chart ----------
try:
    plt.figure(figsize=(6, 4))
    finals = [val_hwas[lr][-1] for lr in lrs]
    plt.bar(np.arange(len(lrs)), finals, tick_label=lrs)
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH Final Validation HWA per Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_final_HWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# ---------- print evaluation summary ----------
for lr in lrs:
    best = max(val_hwas[lr])
    final = val_hwas[lr][-1]
    print(f"LR={lr}: best_HWA={best:.3f}, final_HWA={final:.3f}")

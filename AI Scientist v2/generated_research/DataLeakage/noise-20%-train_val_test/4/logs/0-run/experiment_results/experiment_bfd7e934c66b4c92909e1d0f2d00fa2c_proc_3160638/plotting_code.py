import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["num_epochs"]["SPR_BENCH"]
    epochs_grid = spr_data["tried_epochs"]
    curves = spr_data["epoch_curves"]
    best_val_f1 = spr_data["best_val_f1"]
    test_f1 = spr_data["test_f1"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    curves, epochs_grid, best_val_f1, test_f1 = None, [], [], []

# -------- Plot 1: Loss curves --------
try:
    if curves:
        plt.figure()
        for i, max_ep in enumerate(epochs_grid):
            x = np.arange(1, len(curves["train_loss"][i]) + 1)
            plt.plot(
                x,
                curves["train_loss"][i],
                label=f"train_loss (max_ep={max_ep})",
                linestyle="-",
            )
            plt.plot(
                x,
                curves["val_loss"][i],
                label=f"val_loss (max_ep={max_ep})",
                linestyle="--",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH: Training vs Validation Loss\nLeft: Train, Right: Validation"
        )
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------- Plot 2: F1 curves --------
try:
    if curves:
        plt.figure()
        for i, max_ep in enumerate(epochs_grid):
            x = np.arange(1, len(curves["train_f1"][i]) + 1)
            plt.plot(
                x,
                curves["train_f1"][i],
                label=f"train_F1 (max_ep={max_ep})",
                linestyle="-",
            )
            plt.plot(
                x,
                curves["val_f1"][i],
                label=f"val_F1 (max_ep={max_ep})",
                linestyle="--",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(
            "SPR_BENCH: Training vs Validation Macro-F1\nLeft: Train, Right: Validation"
        )
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# -------- Plot 3: Test F1 vs epochs --------
try:
    if epochs_grid and test_f1:
        plt.figure()
        plt.plot(epochs_grid, test_f1, marker="o")
        for x, y in zip(epochs_grid, test_f1):
            plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
        plt.xlabel("max_epochs tried")
        plt.ylabel("Test Macro-F1")
        plt.title("SPR_BENCH: Test Macro-F1 vs Epoch Budget")
        fname = os.path.join(working_dir, "SPR_BENCH_testF1_vs_epochs.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating Test F1 plot: {e}")
    plt.close()

# -------- Print numeric summary --------
print("Tried max_epochs:", epochs_grid)
print("Best Val Macro-F1 per run:", best_val_f1)
print("Test Macro-F1 per run:", test_f1)

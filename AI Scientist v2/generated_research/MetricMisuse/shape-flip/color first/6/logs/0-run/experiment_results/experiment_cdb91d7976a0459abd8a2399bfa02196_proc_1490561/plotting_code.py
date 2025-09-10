import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load("experiment_data.npy", allow_pickle=True).item()
    lr_data = exp["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_data = {}

# Collect common information
lrs_sorted = sorted(lr_data.keys(), key=lambda x: float(x.replace("e", "e+")))
colors = plt.cm.tab10(np.linspace(0, 1, len(lrs_sorted)))

# 1) Train / Val loss curves in one figure
try:
    plt.figure()
    for clr, lr in zip(colors, lrs_sorted):
        epochs = lr_data[lr]["epochs"]
        plt.plot(
            epochs,
            lr_data[lr]["losses"]["train"],
            color=clr,
            linestyle="-",
            label=f"train lr={lr}",
        )
        plt.plot(
            epochs,
            lr_data[lr]["losses"]["val"],
            color=clr,
            linestyle="--",
            label=f"val lr={lr}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Train vs Val Loss Curves")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# 2) Validation CompWA curves
try:
    plt.figure()
    for clr, lr in zip(colors, lrs_sorted):
        epochs = lr_data[lr]["epochs"]
        plt.plot(
            epochs, lr_data[lr]["metrics"]["val_compwa"], color=clr, label=f"lr={lr}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.title("SPR_BENCH: Validation CompWA over Epochs")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_val_compwa_curves.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating CompWA curve figure: {e}")
    plt.close()

# 3) Test CompWA bar chart
try:
    plt.figure()
    test_scores = [lr_data[lr]["metrics"]["test_compwa"] for lr in lrs_sorted]
    plt.bar(range(len(lrs_sorted)), test_scores, color=colors)
    plt.xticks(range(len(lrs_sorted)), lrs_sorted, rotation=45)
    plt.ylabel("CompWA")
    plt.title("SPR_BENCH: Test CompWA vs Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_test_compwa_bar.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating Test CompWA bar figure: {e}")
    plt.close()

print("Plots saved to:", working_dir)

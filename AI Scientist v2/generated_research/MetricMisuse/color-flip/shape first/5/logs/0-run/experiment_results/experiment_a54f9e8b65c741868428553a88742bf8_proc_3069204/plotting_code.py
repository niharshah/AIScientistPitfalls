import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
spr_key = ("supervised_finetuning_epochs", "SPR_BENCH")
try:
    sweep = experiment_data[spr_key[0]][spr_key[1]]
    epochs_grid = sweep["epochs_grid"]
    train_hsca = sweep["metrics"]["train"]  # list of lists
    val_hsca = sweep["metrics"]["val"]  # list of lists
    test_hsca = sweep["test_hsca"]  # list
except Exception as e:
    print(f"Error extracting data: {e}")
    epochs_grid, train_hsca, val_hsca, test_hsca = [], [], [], []

# ------------------------------------------------------------------
# Plot 1: train / val HSCA curves
try:
    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, len(epochs_grid)))
    for i, max_ep in enumerate(epochs_grid):
        ep_axis = np.arange(1, len(train_hsca[i]) + 1)
        plt.plot(ep_axis, train_hsca[i], color=colors[i], label=f"{max_ep}ep train")
        plt.plot(
            ep_axis,
            val_hsca[i],
            color=colors[i],
            linestyle="--",
            label=f"{max_ep}ep val",
        )
    plt.title("SPR_BENCH – HSCA Curves\nSolid: Train, Dashed: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("HSCA")
    plt.legend(ncol=2, fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_hsca_train_val_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating HSCA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 2: test HSCA vs max epochs
try:
    plt.figure()
    plt.bar([str(e) for e in epochs_grid], test_hsca, color="steelblue")
    plt.title("SPR_BENCH – Test HSCA vs Allowed Fine-tuning Epochs")
    plt.xlabel("Max Fine-tuning Epochs")
    plt.ylabel("Test HSCA")
    for x, y in zip(range(len(epochs_grid)), test_hsca):
        plt.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_test_hsca_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test HSCA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print evaluation metrics
for max_ep, hsca in zip(epochs_grid, test_hsca):
    print(f"Max epochs={max_ep:2d} | Test HSCA={hsca:.4f}")

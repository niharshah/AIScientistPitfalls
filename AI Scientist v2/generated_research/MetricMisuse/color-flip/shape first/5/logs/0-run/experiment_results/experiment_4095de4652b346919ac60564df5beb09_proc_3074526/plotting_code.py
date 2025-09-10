import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# Try to load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------
# Extract metrics for SPR_BENCH
spr_key = "SPR_BENCH"
metrics, losses, test_cwca = {}, {}, None
try:
    metrics = experiment_data[spr_key]["metrics"]
    losses = experiment_data[spr_key]["losses"]
    test_cwca = (
        experiment_data[spr_key]["metrics"]["test"][0]
        if experiment_data[spr_key]["metrics"]["test"]
        else None
    )
    epochs_axis = np.arange(1, len(metrics["train"]) + 1)
except Exception as e:
    print(f"Error extracting metrics: {e}")
    metrics, losses, test_cwca, epochs_axis = {}, {}, None, np.array([])

# ------------------------------------------------------------
# Plot 1: CWCA curves
try:
    if len(epochs_axis):
        plt.figure()
        plt.plot(epochs_axis, metrics["train"], label="Train CWCA", color="steelblue")
        plt.plot(
            epochs_axis,
            metrics["val"],
            label="Validation CWCA",
            color="orange",
            linestyle="--",
        )
        plt.title("SPR_BENCH – CWCA Curves\nSolid: Train, Dashed: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("CWCA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_cwca_train_val_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating CWCA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------
# Plot 2: Loss curves
try:
    if len(epochs_axis):
        plt.figure()
        plt.plot(epochs_axis, losses["train"], label="Train Loss", color="green")
        plt.plot(
            epochs_axis,
            losses["val"],
            label="Validation Loss",
            color="red",
            linestyle="--",
        )
        plt.title("SPR_BENCH – Loss Curves\nSolid: Train, Dashed: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_train_val_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------
# Plot 3: Test CWCA bar
try:
    if test_cwca is not None:
        plt.figure()
        plt.bar(["Test"], [test_cwca], color="purple")
        plt.title("SPR_BENCH – Final Test CWCA")
        plt.ylabel("CWCA")
        plt.ylim(0, 1.0)
        plt.text(0, test_cwca + 0.01, f"{test_cwca:.3f}", ha="center", va="bottom")
        fname = os.path.join(working_dir, "SPR_BENCH_test_cwca_bar.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test CWCA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------
# Print evaluation metric
if test_cwca is not None:
    print(f"Final Test CWCA: {test_cwca:.4f}")
else:
    print("Test CWCA not available.")

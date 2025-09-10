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
    experiment_data = None

if experiment_data:
    exp = experiment_data.get("EmbeddingOnly", {}).get("SPR_BENCH", {})
    losses = exp.get("losses", {})
    metrics = exp.get("metrics", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    val_ccwa = metrics.get("val_CCWA", [])

    # Plot 1: Loss curves
    try:
        plt.figure()
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.title("SPR_BENCH Loss Curves (EmbeddingOnly)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_EmbeddingOnly_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # Plot 2: Validation CCWA
    try:
        plt.figure()
        plt.plot(epochs, val_ccwa, marker="o")
        plt.title("SPR_BENCH Validation CCWA (EmbeddingOnly)")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        fname = os.path.join(working_dir, "SPR_BENCH_EmbeddingOnly_val_CCWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CCWA plot: {e}")
        plt.close()

    # Print final metric
    if val_ccwa:
        print(f"Final Validation CCWA: {val_ccwa[-1]:.4f}")

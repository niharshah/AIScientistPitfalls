import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    exp = experiment_data["batch_size"]["SPR_BENCH"]
    train_losses = exp["losses"]["train"]  # list(list)
    val_losses = exp["losses"]["val"]
    val_hwa = exp["metrics"]["val"]
    batch_sizes = [hp["batch_size"] for hp in exp["hyperparams"]]

    # ---------------- figure 1: loss curves ----------------
    try:
        plt.figure()
        for i, bs in enumerate(batch_sizes):
            epochs = np.arange(1, len(train_losses[i]) + 1)
            plt.plot(epochs, train_losses[i], label=f"train (bs={bs})", linestyle="-")
            plt.plot(epochs, val_losses[i], label=f"val (bs={bs})", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH Loss Curves\nLeft: Train, Right: Validation (Batch-size sweep)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_batch_size.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------------- figure 2: HWA curves ----------------
    try:
        plt.figure()
        for i, bs in enumerate(batch_sizes):
            epochs = np.arange(1, len(val_hwa[i]) + 1)
            plt.plot(epochs, val_hwa[i], label=f"bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title(
            "SPR_BENCH Validation HWA\nLeft: Ground Truth, Right: Model Predictions"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_batch_size.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve plot: {e}")
        plt.close()

    # ---------------- print final metrics ----------------
    print("Final-epoch HWA by batch size:")
    for bs, hwas in zip(batch_sizes, val_hwa):
        print(f"  bs={bs:<3}: {hwas[-1]:.3f}")

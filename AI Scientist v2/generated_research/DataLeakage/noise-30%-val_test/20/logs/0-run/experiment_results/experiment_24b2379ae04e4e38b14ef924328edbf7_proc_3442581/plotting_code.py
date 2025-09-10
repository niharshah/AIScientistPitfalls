import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# paths / loading
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    sweep = experiment_data["num_gru_layers"]["SPR_BENCH"]
    epochs = list(range(1, len(next(iter(sweep.values()))["metrics"]["train"]) + 1))

    # ------------------------------ 1st plot: F1 curves
    try:
        plt.figure()
        for k, res in sweep.items():
            plt.plot(epochs, res["metrics"]["train"], marker="o", label=f"{k} train")
            plt.plot(epochs, res["metrics"]["val"], marker="x", label=f"{k} val")
        plt.title("SPR_BENCH: F1 Score vs Epoch\nGRU Layer Sweep")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # ------------------------------ 2nd plot: Loss curves
    try:
        plt.figure()
        for k, res in sweep.items():
            plt.plot(epochs, res["losses"]["train"], marker="o", label=f"{k} train")
            plt.plot(epochs, res["losses"]["val"], marker="x", label=f"{k} val")
        plt.title("SPR_BENCH: Cross-Entropy Loss vs Epoch\nGRU Layer Sweep")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # ------------------------------ print final metrics
    print("\nFinal Validation F1 Scores:")
    for k, res in sweep.items():
        final_f1 = res["metrics"]["val"][-1]
        print(f"  {k}: {final_f1:.3f}")

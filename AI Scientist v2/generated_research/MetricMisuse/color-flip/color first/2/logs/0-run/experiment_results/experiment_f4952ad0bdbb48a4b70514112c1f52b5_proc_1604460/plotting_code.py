import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ---------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["hidden_dim"]["SPR_BENCH"]
    hidden_dims = sorted(ed["losses"]["train"].keys())
    epochs = len(next(iter(ed["losses"]["train"].values())))  # length of any loss list

    # ----------- Plot 1: loss curves ----------- #
    try:
        plt.figure(figsize=(10, 4))
        plt.suptitle("SPR_BENCH: Loss Curves by Hidden Dimension")
        # Left subplot: training loss
        ax1 = plt.subplot(1, 2, 1)
        for hd in hidden_dims:
            ax1.plot(range(1, epochs + 1), ed["losses"]["train"][hd], label=f"h={hd}")
        ax1.set_title("Left: Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Right subplot: validation loss
        ax2 = plt.subplot(1, 2, 2)
        for hd in hidden_dims:
            ax2.plot(range(1, epochs + 1), ed["losses"]["val"][hd], label=f"h={hd}")
        ax2.set_title("Right: Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------- Plot 2: test metrics ----------- #
    try:
        metrics_names = ["CWA", "SWA", "GCWA"]
        bar_width = 0.2
        x = np.arange(len(hidden_dims))

        plt.figure(figsize=(8, 5))
        for i, m in enumerate(metrics_names):
            vals = [ed["metrics"]["test"][hd][m] for hd in hidden_dims]
            plt.bar(x + i * bar_width, vals, width=bar_width, label=m)

        plt.title("SPR_BENCH: Test Weighted Accuracies by Hidden Dimension")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Accuracy")
        plt.xticks(x + bar_width, [str(hd) for hd in hidden_dims])
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating metrics bar plot: {e}")
        plt.close()

    # ----------- Print test metrics table ----------- #
    print("\nFinal Test Metrics")
    print("hd\tCWA\tSWA\tGCWA")
    for hd in hidden_dims:
        m = ed["metrics"]["test"][hd]
        print(f'{hd}\t{m["CWA"]:.3f}\t{m["SWA"]:.3f}\t{m["GCWA"]:.3f}')

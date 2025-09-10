import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# -------- load data --------
try:
    edict = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = edict["RandomClusterAssignment"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

if run is not None:
    epochs = range(1, len(run["losses"]["train"]) + 1)

    # -------- 1) Loss curve --------
    try:
        plt.figure()
        plt.plot(epochs, run["losses"]["train"], label="Train")
        plt.plot(epochs, run["losses"]["val"], label="Validation")
        plt.title("SPR_BENCH Loss vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------- 2) Validation metrics over epochs --------
    try:
        val_metrics = run["metrics"]["val"]
        keys = ["acc", "cwa", "swa", "ccwa"]
        for k in keys:
            plt.plot(epochs, [m[k] for m in val_metrics], label=k.upper())
        plt.title("SPR_BENCH Validation Metrics over Epochs\n(ACC, CWA, SWA, CCWA)")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation metric plot: {e}")
        plt.close()

    # -------- 3) Test set summary --------
    try:
        test = run["metrics"]["test"]
        plt.figure()
        bars = plt.bar(list(test.keys()), list(test.values()), color="skyblue")
        for b in bars:
            plt.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.title("SPR_BENCH Test Metrics (ACC / CWA / SWA / CCWA)")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar chart: {e}")
        plt.close()

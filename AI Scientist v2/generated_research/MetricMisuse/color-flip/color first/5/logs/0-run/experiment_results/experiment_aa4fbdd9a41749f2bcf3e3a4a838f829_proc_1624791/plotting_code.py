import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = exp["learning_rate"]["SPR_BENCH"]["runs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = []


# Helper to get epochs
def get_epochs(run):
    return np.arange(1, len(run["losses"]["train"]) + 1)


# --------- Figure 1 : Loss curves -------------
try:
    plt.figure()
    for run in runs:
        epochs = get_epochs(run)
        plt.plot(
            epochs,
            run["losses"]["train"],
            label=f"lr={run['lr']:.0e} train",
            linestyle="-",
        )
        plt.plot(
            epochs,
            run["losses"]["val"],
            label=f"lr={run['lr']:.0e} val",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# --------- Figure 2 : Accuracy curves ----------
try:
    plt.figure()
    for run in runs:
        epochs = get_epochs(run)
        tr_acc = [m["acc"] for m in run["metrics"]["train"]]
        val_acc = [m["acc"] for m in run["metrics"]["val"]]
        plt.plot(epochs, tr_acc, label=f"lr={run['lr']:.0e} train", linestyle="-")
        plt.plot(epochs, val_acc, label=f"lr={run['lr']:.0e} val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Accuracy Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# --------- Figure 3 : Test metrics bar chart ----
try:
    metrics = ["acc", "cwa", "swa", "compwa"]
    x = np.arange(len(metrics))
    width = 0.15
    plt.figure()
    for i, run in enumerate(runs):
        vals = [run["metrics"]["test"][m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=f"lr={run['lr']:.0e}")
    plt.xticks(x + width * (len(runs) - 1) / 2, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("SPR_BENCH – Test Metrics by Learning Rate")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics plot: {e}")
    plt.close()

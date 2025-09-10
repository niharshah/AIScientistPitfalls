import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    bs_data = experiment_data.get("batch_size_tuning", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    bs_data = {}


# Helper to pretty-print the final test metrics
def print_test_metrics(bs_key, m):
    print(
        f"{bs_key:>6} | "
        f"Acc {m['acc']:.3f} | "
        f"SWA {m['swa']:.3f} | "
        f"CWA {m['cwa']:.3f} | "
        f"NRGS {m['nrgs']:.3f}"
    )


# --------------------------------------------------------------
# 1 plot per batch size: training / validation loss curves
for i, (bs_key, log) in enumerate(
    sorted(bs_data.items(), key=lambda x: int(x[0].split("_")[1]))
):
    try:
        epochs = list(range(1, len(log["losses"]["train"]) + 1))
        train_loss = log["losses"]["train"]
        val_loss = log["losses"]["val"]
        val_acc = [m["acc"] for m in log["metrics"]["val"]]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
        ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")

        # secondary axis for accuracy
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_acc, label="Val Acc", color="tab:green", linestyle="--")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")

        plt.title(f"SPR_BENCH Loss & ValAcc Curves – {bs_key}")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"spr_bench_{bs_key}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {bs_key}: {e}")
        plt.close()

# --------------------------------------------------------------
# Aggregate final test metrics across batch sizes
try:
    bs_keys = sorted(bs_data.keys(), key=lambda x: int(x.split("_")[1]))
    metrics = ["acc", "swa", "cwa", "nrgs"]
    vals = {m: [bs_data[k]["metrics"]["test"][m] for k in bs_keys] for m in metrics}

    x = np.arange(len(bs_keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width - 1.5 * width, vals[m], width, label=m.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(bs_keys)
    ax.set_ylim(0, 1)
    ax.set_title("SPR_BENCH Test Metrics vs Batch Size")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_test_metrics_across_bs.png")
    plt.savefig(fname)
    plt.close()

    print("\nFinal Test Metrics:")
    for k in bs_keys:
        print_test_metrics(k, bs_data[k]["metrics"]["test"])
except Exception as e:
    print(f"Error creating aggregated metrics plot: {e}")
    plt.close()

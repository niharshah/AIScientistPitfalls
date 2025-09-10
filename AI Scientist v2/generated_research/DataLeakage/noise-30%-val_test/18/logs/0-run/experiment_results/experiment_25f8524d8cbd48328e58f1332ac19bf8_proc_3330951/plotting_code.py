import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Extract once for convenience
bench = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})

# --- Plot 1: Macro-F1 curves -------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    for i, (bs, log) in enumerate(bench.items()):
        epochs = log["epochs"]
        plt.subplot(1, 2, 1)
        plt.plot(epochs, log["metrics"]["train"], label=f"bs={bs}")
        plt.title("Train Macro-F1 (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.subplot(1, 2, 2)
        plt.plot(epochs, log["metrics"]["val"], label=f"bs={bs}")
        plt.title("Val Macro-F1 (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
    for ax in plt.gcf().axes:
        ax.legend()
    plt.suptitle("Left: Training, Right: Validation Macro-F1")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# --- Plot 2: Loss curves -----------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    for bs, log in bench.items():
        epochs = log["epochs"]
        plt.subplot(1, 2, 1)
        plt.plot(epochs, log["losses"]["train"], label=f"bs={bs}")
        plt.title("Train Loss (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.subplot(1, 2, 2)
        plt.plot(epochs, log["losses"]["val"], label=f"bs={bs}")
        plt.title("Val Loss (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
    for ax in plt.gcf().axes:
        ax.legend()
    plt.suptitle("Left: Training, Right: Validation Loss")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss curves: {e}")
    plt.close()

# --- Plot 3: Dev vs Test best Macro-F1 ---------------------------------------
try:
    bs_list, dev_best, test_f1 = [], [], []
    for bs, log in bench.items():
        bs_list.append(int(bs))
        dev_best.append(max(log["metrics"]["val"]))
        test_f1.append(log["test_f1"])
    x = np.arange(len(bs_list))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, dev_best, width, label="Best Dev F1")
    plt.bar(x + width / 2, test_f1, width, label="Test F1")
    plt.xticks(x, bs_list)
    plt.xlabel("Batch Size")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Dev vs Test Macro-F1 by Batch Size")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_dev_test_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

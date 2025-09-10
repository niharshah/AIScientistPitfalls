import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- SETUP ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bs_data = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})
batch_sizes = sorted(map(int, bs_data.keys()))
if not batch_sizes:
    print("No data found for plotting.")
    exit()

# reusable colors
colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

# ---------- FIGURE 1: Train / Val Accuracy ----------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, bs in enumerate(batch_sizes):
        metrics = bs_data[str(bs)]["metrics"]
        epochs = np.arange(1, len(metrics["train_acc"]) + 1)
        axes[0].plot(epochs, metrics["train_acc"], label=f"bs={bs}", color=colors[i])
        axes[1].plot(epochs, metrics["val_acc"], label=f"bs={bs}", color=colors[i])
    axes[0].set_title("Train Acc")
    axes[1].set_title("Val Acc")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    fig.suptitle("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_train_val_accuracy.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy figure: {e}")
    plt.close()

# ---------- FIGURE 2: Train Loss ----------
try:
    plt.figure()
    for i, bs in enumerate(batch_sizes):
        losses = bs_data[str(bs)]["losses"]["train"]
        plt.plot(range(1, len(losses) + 1), losses, label=f"bs={bs}", color=colors[i])
    plt.title("SPR_BENCH Train Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_train_loss.png")
    plt.tight_layout()
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss figure: {e}")
    plt.close()


# helper to gather final metrics
def gather_metric(key):
    return [
        bs_data[str(bs)][key] if key in bs_data[str(bs)] else np.nan
        for bs in batch_sizes
    ]


# ---------- FIGURE 3: Test Accuracy vs Batch Size ----------
try:
    test_accs = gather_metric("test_acc")
    plt.figure()
    plt.plot(batch_sizes, test_accs, marker="o")
    plt.title("SPR_BENCH Test Accuracy vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    fname = os.path.join(working_dir, "spr_bench_test_accuracy.png")
    plt.tight_layout()
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy figure: {e}")
    plt.close()

# ---------- FIGURE 4: Fidelity vs Batch Size ----------
try:
    fidelities = gather_metric("fidelity")
    plt.figure()
    plt.plot(batch_sizes, fidelities, marker="s", color="orange")
    plt.title("SPR_BENCH Rule Fidelity vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Fidelity")
    plt.grid(True)
    fname = os.path.join(working_dir, "spr_bench_fidelity.png")
    plt.tight_layout()
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating fidelity figure: {e}")
    plt.close()

# ---------- FIGURE 5: FAGM vs Batch Size ----------
try:
    fagms = gather_metric("fagm")
    plt.figure()
    plt.plot(batch_sizes, fagms, marker="^", color="red")
    plt.title("SPR_BENCH FAGM vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("FAGM")
    plt.grid(True)
    fname = os.path.join(working_dir, "spr_bench_fagm.png")
    plt.tight_layout()
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating FAGM figure: {e}")
    plt.close()

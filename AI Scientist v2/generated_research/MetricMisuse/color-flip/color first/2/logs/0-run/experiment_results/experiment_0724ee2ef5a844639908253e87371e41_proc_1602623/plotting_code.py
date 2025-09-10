import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

data_branch = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})
if not data_branch:
    print("No SPR_BENCH data found in experiment_data.npy")
    exit(0)

batch_sizes = sorted(data_branch.keys())
epochs = max(len(data_branch[bs]["losses"]["train"]) for bs in batch_sizes)

# -------- gather arrays --------
train_losses = {bs: data_branch[bs]["losses"]["train"] for bs in batch_sizes}
val_losses = {bs: data_branch[bs]["losses"]["val"] for bs in batch_sizes}
test_metrics = {bs: data_branch[bs]["metrics"]["test"] for bs in batch_sizes}

# -------- print test metrics --------
print("Final test metrics (CWA | SWA | GCWA):")
for bs in batch_sizes:
    m = test_metrics[bs]
    print(f'  BS={bs:>3}: {m["CWA"]:.3f} | {m["SWA"]:.3f} | {m["GCWA"]:.3f}')


# -------- plotting helpers --------
def save_fig(fig, name):
    path = os.path.join(working_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# -------- Figure 1: training loss --------
try:
    fig1 = plt.figure()
    for bs in batch_sizes:
        plt.plot(
            range(1, len(train_losses[bs]) + 1), train_losses[bs], label=f"BS={bs}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("SPR_BENCH – Training Loss vs Epoch")
    plt.legend()
    save_fig(fig1, "SPR_BENCH_training_loss.png")
except Exception as e:
    print(f"Error creating training-loss plot: {e}")
    plt.close()

# -------- Figure 2: validation loss --------
try:
    fig2 = plt.figure()
    for bs in batch_sizes:
        plt.plot(range(1, len(val_losses[bs]) + 1), val_losses[bs], label=f"BS={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("SPR_BENCH – Validation Loss vs Epoch")
    plt.legend()
    save_fig(fig2, "SPR_BENCH_validation_loss.png")
except Exception as e:
    print(f"Error creating validation-loss plot: {e}")
    plt.close()

# -------- Figure 3: test metrics bar chart --------
try:
    metrics_names = ["CWA", "SWA", "GCWA"]
    x = np.arange(len(batch_sizes))
    width = 0.25
    fig3, ax = plt.subplots()
    for i, mname in enumerate(metrics_names):
        vals = [test_metrics[bs][mname] for bs in batch_sizes]
        ax.bar(x + i * width - width, vals, width, label=mname)
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Score")
    ax.set_title("SPR_BENCH – Test Metrics by Batch Size")
    ax.legend()
    save_fig(fig3, "SPR_BENCH_test_metrics.png")
except Exception as e:
    print(f"Error creating test-metrics plot: {e}")
    plt.close()

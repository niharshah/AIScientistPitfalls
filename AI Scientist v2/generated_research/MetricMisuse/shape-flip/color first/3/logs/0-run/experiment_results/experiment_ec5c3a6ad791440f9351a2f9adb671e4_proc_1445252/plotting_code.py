import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------- #
# load data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit


# helper to fetch nested dict safely
def get_batch_runs():
    try:
        return experiment_data["batch_size"]["SPR_BENCH"]
    except KeyError:
        print("Unexpected experiment_data structure.")
        return {}


runs = get_batch_runs()
if not runs:
    raise SystemExit

batch_sizes = sorted(int(k) for k in runs.keys())
colors = {bs: c for bs, c in zip(batch_sizes, ["r", "g", "b", "c", "m"])}

# ---------------------------------------------------------------------- #
# 1) Train/Val BWA curves
try:
    plt.figure()
    for bs in batch_sizes:
        data = runs[str(bs)]
        epochs = np.arange(1, len(data["metrics"]["train"]) + 1)
        plt.plot(
            epochs,
            data["metrics"]["train"],
            "--",
            color=colors[bs],
            label=f"train bs={bs}",
        )
        plt.plot(
            epochs, data["metrics"]["val"], "-", color=colors[bs], label=f"val bs={bs}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("BWA")
    plt.title("SPR_BENCH: Train vs. Validation BWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_val_BWA.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating BWA plot: {e}")
    plt.close()

# ---------------------------------------------------------------------- #
# 2) Train/Val Loss curves
try:
    plt.figure()
    for bs in batch_sizes:
        data = runs[str(bs)]
        epochs = np.arange(1, len(data["losses"]["train"]) + 1)
        plt.plot(
            epochs,
            data["losses"]["train"],
            "--",
            color=colors[bs],
            label=f"train bs={bs}",
        )
        plt.plot(
            epochs, data["losses"]["val"], "-", color=colors[bs], label=f"val bs={bs}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------------------------------------------------------------- #
# 3) Test metrics bar chart
try:
    metrics = ["bwa", "cwa", "swa"]
    x = np.arange(len(batch_sizes))
    width = 0.25
    plt.figure()
    for i, m in enumerate(metrics):
        vals = [runs[str(bs)]["test_metrics"][m] for bs in batch_sizes]
        plt.bar(x + i * width, vals, width, label=m.upper())
    plt.xticks(x + width, [str(bs) for bs in batch_sizes])
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Test Metrics by Batch Size")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics plot: {e}")
    plt.close()

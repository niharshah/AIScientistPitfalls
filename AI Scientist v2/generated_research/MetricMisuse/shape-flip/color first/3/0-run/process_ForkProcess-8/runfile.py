import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
spr_runs = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})
batch_sizes = sorted(spr_runs.keys(), key=lambda x: int(x))[:4]  # ensure only 4
fig_count = 0

for bs in batch_sizes:
    try:
        run = spr_runs[bs]
        epochs = np.arange(1, len(run["metrics"]["train"]) + 1)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Loss curves (left y-axis)
        ax1.plot(epochs, run["losses"]["train"], "r--", label="Train Loss")
        ax1.plot(epochs, run["losses"]["val"], "r-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss", color="r")
        ax1.tick_params(axis="y", labelcolor="r")

        # BWA curves (right y-axis)
        ax2.plot(epochs, run["metrics"]["train"], "b--", label="Train BWA")
        ax2.plot(epochs, run["metrics"]["val"], "b-", label="Val BWA")
        ax2.set_ylabel("Balanced Weighted Accuracy", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

        plt.title(f"SPR_BENCH (bs={bs})\nLeft: Loss, Right: BWA – SPR_BENCH")
        fig.tight_layout()
        name = f"SPR_BENCH_bs{bs}_loss_bwa_curves.png"
        path = os.path.join(working_dir, name)
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating curves for bs={bs}: {e}")
        plt.close()

# ------------------------------------------------------------------
# Summary bar chart of final test BWA across batch sizes
try:
    final_bwa = [spr_runs[bs]["metrics"]["val"][-1] for bs in batch_sizes]
    x = np.arange(len(batch_sizes))
    plt.figure()
    plt.bar(x, final_bwa, color="skyblue")
    plt.xticks(x, batch_sizes)
    plt.ylabel("Final Dev BWA")
    plt.title("SPR_BENCH – Final Balanced Weighted Accuracy by Batch Size")
    plt.tight_layout()
    name = "SPR_BENCH_final_bwa_vs_batch_size.png"
    path = os.path.join(working_dir, name)
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
    fig_count += 1
except Exception as e:
    print(f"Error creating summary bar chart: {e}")
    plt.close()

print(f"Total figures created: {fig_count}")

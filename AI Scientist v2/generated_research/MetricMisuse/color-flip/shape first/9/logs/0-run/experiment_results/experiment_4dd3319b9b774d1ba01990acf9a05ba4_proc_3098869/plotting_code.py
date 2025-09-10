import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper to fetch data safely
bench_key = ("batch_size_tuning", "SPR_BENCH")
runs = (
    experiment_data.get(bench_key[0], {}).get(bench_key[1], {})
    if experiment_data
    else {}
)
batch_sizes = sorted(runs.keys(), key=lambda x: int(x))  # ['32','64',...]

# -------- 1) training loss curves --------
try:
    plt.figure()
    for bs in batch_sizes:
        y = runs[bs]["losses"]["train"]
        plt.plot(range(1, len(y) + 1), y, label=f"bs={bs}")
    plt.title("SPR_BENCH – Training Loss vs Epoch (all batch sizes)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_loss_all_bs.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating training‐loss figure: {e}")
    plt.close()

# -------- 2) validation loss curves --------
try:
    plt.figure()
    for bs in batch_sizes:
        y = runs[bs]["losses"]["val"]
        plt.plot(range(1, len(y) + 1), y, label=f"bs={bs}")
    plt.title("SPR_BENCH – Validation Loss vs Epoch (all batch sizes)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_loss_all_bs.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating validation‐loss figure: {e}")
    plt.close()

# -------- 3) validation CWA-2D curves --------
try:
    plt.figure()
    for bs in batch_sizes:
        y = runs[bs]["metrics"]["val"]
        plt.plot(range(1, len(y) + 1), y, label=f"bs={bs}")
    plt.title("SPR_BENCH – Validation CWA-2D vs Epoch (all batch sizes)")
    plt.xlabel("Epoch")
    plt.ylabel("CWA-2D")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_cwa_all_bs.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating CWA curve figure: {e}")
    plt.close()

# -------- 4) final-epoch CWA bar chart --------
try:
    plt.figure()
    final_cwas = [runs[bs]["metrics"]["val"][-1] for bs in batch_sizes]
    plt.bar([int(bs) for bs in batch_sizes], final_cwas, color="steelblue")
    plt.title("SPR_BENCH – Final Epoch CWA-2D by Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Final CWA-2D")
    fname = os.path.join(working_dir, "SPR_BENCH_final_cwa_bar.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating final CWA bar figure: {e}")
    plt.close()

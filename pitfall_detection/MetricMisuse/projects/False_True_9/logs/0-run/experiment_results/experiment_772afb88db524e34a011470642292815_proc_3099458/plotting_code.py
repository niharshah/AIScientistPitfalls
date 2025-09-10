import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    sweep = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})
    batch_sizes = sorted([int(bs) for bs in sweep.keys()])
    # containers for final metrics
    final_hwa = {}

    # ------------ 1) loss curves -------------
    try:
        plt.figure()
        for bs in batch_sizes:
            bs_str = str(bs)
            train = sweep[bs_str]["losses"]["train"]  # [(epoch, loss), ...]
            val = sweep[bs_str]["losses"]["val"]
            epochs_t, losses_t = zip(*train)
            epochs_v, losses_v = zip(*val)
            plt.plot(epochs_t, losses_t, linestyle="-", label=f"train bs={bs}")
            plt.plot(epochs_v, losses_v, linestyle="--", label=f"val bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss\nBatch Size Sweep")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------ 2) validation HWA curves -------------
    try:
        plt.figure()
        for bs in batch_sizes:
            bs_str = str(bs)
            metrics = sweep[bs_str]["metrics"]["val"]  # [(epoch, swa, cwa, hwa), ...]
            epochs, _, _, hwas = zip(*metrics)
            plt.plot(epochs, hwas, label=f"HWA bs={bs}")
            # store final epoch hwa
            final_hwa[bs] = hwas[-1]
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("SPR_BENCH: Validation HWA\nBatch Size Sweep")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve plot: {e}")
        plt.close()

    # ------------ 3) final HWA per batch size -------------
    try:
        plt.figure()
        bs_list = list(final_hwa.keys())
        hwa_vals = [final_hwa[bs] for bs in bs_list]
        plt.bar([str(bs) for bs in bs_list], hwa_vals, color="skyblue")
        plt.xlabel("Batch Size")
        plt.ylabel("Final Epoch HWA")
        plt.title("SPR_BENCH: Final Harmonic Weighted Accuracy\nvs Batch Size")
        fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final HWA bar plot: {e}")
        plt.close()

    # ------------ print summary -------------
    print("Final HWA by batch size:")
    for bs in batch_sizes:
        print(f"  bs={bs}: HWA={final_hwa.get(bs, 'N/A'):.4f}")

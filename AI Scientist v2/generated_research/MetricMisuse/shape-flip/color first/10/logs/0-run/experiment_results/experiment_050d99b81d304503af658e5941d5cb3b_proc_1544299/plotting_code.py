import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data["batch_size"]["SPR_BENCH"]
    bs_keys = sorted(runs.keys(), key=lambda x: int(x))  # ['32','64','128']
    # --------- PLOT 1: train/val loss ----------
    try:
        plt.figure()
        for bs in bs_keys:
            tr = runs[bs]["losses"]["train"]
            va = runs[bs]["losses"]["val"]
            epochs = list(range(1, len(tr) + 1))
            plt.plot(epochs, tr, label=f"train bs={bs}")
            plt.plot(epochs, va, "--", label=f"val bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------- PLOT 2: validation HWA ----------
    try:
        plt.figure()
        for bs in bs_keys:
            hwa = [m["hwa"] for m in runs[bs]["metrics"]["val"]]
            epochs = list(range(1, len(hwa) + 1))
            plt.plot(epochs, hwa, label=f"bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("SPR_BENCH: Validation HWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_hwa.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # --------- PLOT 3: test HWA bar chart ----------
    try:
        plt.figure()
        hwa_vals = [runs[bs]["metrics"]["test"]["hwa"] for bs in bs_keys]
        plt.bar(bs_keys, hwa_vals)
        plt.xlabel("Batch size")
        plt.ylabel("Test HWA")
        plt.title("SPR_BENCH: Test HWA vs Batch Size")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_hwa_bars.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test HWA plot: {e}")
        plt.close()

    # --------- print final test metrics ----------
    print("Final test metrics per batch size")
    print("bs\tCWA\tSWA\tHWA")
    for bs in bs_keys:
        m = runs[bs]["metrics"]["test"]
        print(f"{bs}\t{m['cwa']:.3f}\t{m['swa']:.3f}\t{m['hwa']:.3f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ----- paths -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    lr_dict = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
    lrs = list(lr_dict.keys())

    # ---------- 1. loss curves ----------
    try:
        plt.figure()
        for lr in lrs:
            losses = lr_dict[lr]["losses"]
            epochs = np.arange(len(losses["train"]))
            # subsample to at most 5 points
            idx = np.linspace(0, len(epochs) - 1, min(5, len(epochs))).astype(int)
            plt.plot(
                epochs[idx],
                np.array(losses["train"])[idx],
                marker="o",
                label=f"train lr={lr}",
            )
            plt.plot(
                epochs[idx],
                np.array(losses["val"])[idx],
                marker="x",
                linestyle="--",
                label=f"val lr={lr}",
            )
        plt.title("SPR_BENCH Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- 2. validation HWA ----------
    try:
        plt.figure()
        for lr in lrs:
            hwa_vals = [m["HWA"] for m in lr_dict[lr]["metrics"]["val"]]
            epochs = np.arange(len(hwa_vals))
            idx = np.linspace(0, len(epochs) - 1, min(5, len(epochs))).astype(int)
            plt.plot(
                epochs[idx],
                np.array(hwa_vals)[idx],
                marker="s",
                label=f"val HWA lr={lr}",
            )
        plt.title("SPR_BENCH Validation HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_HWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ---------- 3. test HWA bar chart ----------
    try:
        plt.figure()
        test_hwa = [lr_dict[lr]["test_metrics"]["HWA"] for lr in lrs]
        plt.bar(np.arange(len(lrs)), test_hwa, tick_label=lrs)
        plt.title("SPR_BENCH Final Test HWA per Learning Rate")
        plt.ylabel("HWA")
        fname = os.path.join(working_dir, "SPR_BENCH_test_HWA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test HWA bar chart: {e}")
        plt.close()

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

# Helper: collect runs
runs = sorted(
    [k for k in experiment_data.keys() if k.startswith("weight_decay_")],
    key=lambda s: float(s.split("_")[-1]),
)

# Figure 1: loss curves
try:
    plt.figure(figsize=(10, 4))
    for rk in runs:
        tr = experiment_data[rk]["SPR_BENCH"]["losses"]["train"]
        vl = experiment_data[rk]["SPR_BENCH"]["losses"]["val"]
        epochs = np.arange(1, len(tr) + 1)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr, label=rk)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.subplot(1, 2, 2)
        plt.plot(epochs, vl, label=rk)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Val Loss")
    plt.subplot(1, 2, 1)
    plt.legend(fontsize=7)
    plt.suptitle("SPR_BENCH Loss Curves (Left: Train, Right: Val)")
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# Figure 2: validation HWA curves
try:
    plt.figure()
    for rk in runs:
        hwa = [m["hwa"] for m in experiment_data[rk]["SPR_BENCH"]["metrics"]["val"]]
        epochs = np.arange(1, len(hwa) + 1)
        plt.plot(epochs, hwa, marker="o", label=rk)
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH Validation HWA vs Epoch")
    plt.legend(fontsize=7)
    save_path = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve figure: {e}")
    plt.close()

# Figure 3: test HWA bar chart
try:
    plt.figure()
    wd_vals = [float(r.split("_")[-1]) for r in runs]
    hwa_test = [experiment_data[r]["SPR_BENCH"]["metrics"]["test"]["hwa"] for r in runs]
    plt.bar(range(len(wd_vals)), hwa_test, tick_label=wd_vals)
    plt.xlabel("Weight Decay")
    plt.ylabel("Test HWA")
    plt.title("SPR_BENCH Test HWA for different Weight Decay")
    save_path = os.path.join(working_dir, "SPR_BENCH_test_HWA_bars.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()
except Exception as e:
    print(f"Error creating test HWA bar figure: {e}")
    plt.close()

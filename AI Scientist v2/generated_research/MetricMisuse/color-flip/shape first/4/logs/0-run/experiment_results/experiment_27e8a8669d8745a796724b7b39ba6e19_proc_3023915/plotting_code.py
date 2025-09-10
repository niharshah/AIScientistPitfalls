import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Ensure working directory exists
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_path = os.path.join(os.getcwd(), "experiment_data.npy")
    if not os.path.isfile(experiment_path):  # fallback to working_dir
        experiment_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(experiment_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

hid_runs = experiment_data.get("hid_dim", {}) if experiment_data else {}

# Early exit if nothing to plot
if not hid_runs:
    print("No data found to plot.")
else:
    # ---------- collect data ----------
    hids = sorted(
        hid_runs.keys(), key=lambda x: int(x[1:])
    )  # e.g. ['h64', 'h128', ...]
    epochs = len(next(iter(hid_runs.values()))["SPR_BENCH"]["losses"]["train"])

    # ---------- 1. Loss curves 2x2 grid ----------
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        for ax, hid in zip(axes, hids):
            log = hid_runs[hid]["SPR_BENCH"]
            ax.plot(range(1, epochs + 1), log["losses"]["train"], label="Train")
            ax.plot(range(1, epochs + 1), log["losses"]["val"], label="Val")
            ax.set_title(f"{hid} Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy")
            ax.legend()
        fig.suptitle("SPR_BENCH: Training vs. Validation Loss per Hidden Dim")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- 2. Validation HWA over epochs ----------
    try:
        plt.figure()
        for hid in hids:
            log = hid_runs[hid]["SPR_BENCH"]
            plt.plot(range(1, epochs + 1), log["metrics"]["val"], label=hid)
        plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_val_HWA.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve plot: {e}")
        plt.close()

    # ---------- 3. Final epoch HWA bar chart ----------
    try:
        plt.figure()
        final_hwa = [hid_runs[hid]["SPR_BENCH"]["metrics"]["val"][-1] for hid in hids]
        plt.bar(hids, final_hwa)
        plt.title("SPR_BENCH: Final Validation HWA by Hidden Dim")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Final HWA")
        save_path = os.path.join(working_dir, "SPR_BENCH_final_HWA_bar.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA bar plot: {e}")
        plt.close()

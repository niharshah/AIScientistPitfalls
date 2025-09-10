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
    experiment_data = {}


# helper to get lists
def _get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


spr_runs = _get(experiment_data, "num_epochs_tuning", "SPR_BENCH", default={})

# -------------- FIGURE 1 -------------------
try:
    if spr_runs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for epochs_str, logs in spr_runs.items():
            tr_loss = logs["losses"]["train"]
            val_loss = logs["losses"]["val"]
            epochs = range(1, len(tr_loss) + 1)
            axes[0].plot(epochs, tr_loss, label=f"{epochs_str}e")
            axes[1].plot(epochs, val_loss, label=f"{epochs_str}e")
        axes[0].set_title("Left: Train Loss")
        axes[1].set_title("Right: Validation Loss")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.suptitle("SPR_BENCH Training vs Validation Loss Curves")
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# -------------- FIGURE 2 -------------------
try:
    if spr_runs:
        plt.figure(figsize=(6, 4))
        for epochs_str, logs in spr_runs.items():
            hwa = [m["hwa"] for m in logs["metrics"]["val"]]
            epochs = range(1, len(hwa) + 1)
            plt.plot(epochs, hwa, label=f"{epochs_str}e")
        plt.title("SPR_BENCH Validation HWA Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves plot: {e}")
    plt.close()

# -------------- FIGURE 3 -------------------
try:
    if spr_runs:
        best_hwa = {
            int(k): max(m["hwa"] for m in v["metrics"]["val"])
            for k, v in spr_runs.items()
        }
        xs, ys = zip(*sorted(best_hwa.items()))
        plt.figure(figsize=(6, 4))
        plt.bar([str(x) for x in xs], ys)
        plt.title("SPR_BENCH Best HWA vs Max Epochs")
        plt.xlabel("Max Epochs")
        plt.ylabel("Best HWA")
        save_path = os.path.join(working_dir, "SPR_BENCH_best_HWA_vs_epochs.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
    plt.close()

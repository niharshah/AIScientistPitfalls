import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- load data -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------- locate runs -------------------
runs = (
    experiment_data.get("random_token_mask_15", {}).get("SPR_BENCH", {}).get("runs", {})
)

# ------------------- plotting -------------------
max_figures = 5
for i, (run_key, run_val) in enumerate(runs.items()):
    if i >= max_figures:
        print("Reached maximum number of figures (5); skipping remaining runs.")
        break
    try:
        # ---- prepare data ----
        tr_loss = run_val["losses"]["train"]
        val_loss = run_val["losses"]["val"]
        tr_hwa = [m[2] for m in run_val["metrics"]["train"]]
        val_hwa = [m[2] for m in run_val["metrics"]["val"]]
        epochs = list(range(1, len(tr_loss) + 1))

        # ---- plot ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Left panel: Loss
        axes[0].plot(epochs, tr_loss, label="Train")
        axes[0].plot(epochs, val_loss, label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()

        # Right panel: HWA
        axes[1].plot(epochs, tr_hwa, label="Train")
        axes[1].plot(epochs, val_hwa, label="Val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("HWA")
        axes[1].set_title("Harmonic Weighted Accuracy")
        axes[1].legend()

        fig.suptitle(f"SPR_BENCH {run_key} - Left: Loss, Right: HWA", fontsize=12)
        fname = f"spr_bench_{run_key}_loss_hwa.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close(fig)
        print(f"Saved plot for {run_key} to {fname}")
    except Exception as e:
        print(f"Error creating plot for {run_key}: {e}")
        plt.close()

# ------------------- print final test metrics -------------------
for run_key, run_val in runs.items():
    swa, cwa, hwa = run_val["metrics"]["test"]
    print(f"{run_key}  |  Test SWA: {swa:.4f}  CWA: {cwa:.4f}  HWA: {hwa:.4f}")

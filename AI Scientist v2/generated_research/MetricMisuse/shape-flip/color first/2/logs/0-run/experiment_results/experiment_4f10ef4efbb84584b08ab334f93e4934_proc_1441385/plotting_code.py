import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    lr_section = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
    final_dwa = {}

    # ---------- Plot 1: Train/Val Loss ----------
    try:
        plt.figure()
        for lr_key, run in lr_section.items():
            train_losses = [v for _, v in run["losses"]["train"]]
            val_losses = [v for _, v in run["losses"]["val"]]
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, label=f"{lr_key} train")
            plt.plot(epochs, val_losses, "--", label=f"{lr_key} val")
        plt.title("SPR_BENCH: Training & Validation Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- Plot 2: Validation DWA ----------
    try:
        plt.figure()
        for lr_key, run in lr_section.items():
            dwa_vals = [v for _, v in run["metrics"]["val"]]
            epochs = range(1, len(dwa_vals) + 1)
            plt.plot(epochs, dwa_vals, label=lr_key)
            final_dwa[lr_key] = dwa_vals[-1] if dwa_vals else np.nan
        plt.title("SPR_BENCH: Dual-Weighted Accuracy (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("DWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_DWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating DWA plot: {e}")
        plt.close()

    # ---------- Plot 3: Final DWA Bar ----------
    try:
        plt.figure()
        keys = list(final_dwa.keys())
        vals = [final_dwa[k] for k in keys]
        plt.bar(keys, vals, color="skyblue")
        plt.title("SPR_BENCH: Final Dual-Weighted Accuracy by Learning Rate")
        plt.ylabel("Final DWA")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_DWA_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot: {e}")
        plt.close()

    # ---------- Print evaluation summary ----------
    print("\nFinal Validation DWA per learning rate:")
    for k, v in final_dwa.items():
        print(f"  {k}: {v:.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("num_epochs", {})

# ---------- per-run loss curves ----------
for run_key, run_val in runs.items():
    try:
        data = run_val.get("SPR_BENCH", {})
        train_loss = [v for _, v in data.get("losses", {}).get("train", [])]
        val_loss = [v for _, v in data.get("losses", {}).get("val", [])]
        epochs = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.title(f"SPR_BENCH Loss Curves – {run_key} Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"SPR_BENCH_loss_curves_{run_key}_epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {run_key} epochs: {e}")
        plt.close()

# ---------- aggregated DWA curves ----------
try:
    plt.figure()
    for run_key, run_val in runs.items():
        data = run_val.get("SPR_BENCH", {})
        dwa_vals = [v for _, v in data.get("metrics", {}).get("val", [])]
        epochs = range(1, len(dwa_vals) + 1)
        plt.plot(epochs, dwa_vals, label=f"{run_key} Epochs")
    plt.title("SPR_BENCH Validation Dual Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("DWA")
    plt.legend()
    fname = "SPR_BENCH_DWA_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating DWA plot: {e}")
    plt.close()

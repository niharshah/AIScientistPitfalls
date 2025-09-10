import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to fetch data safely
def get_run_record(variant, epochs_str):
    return experiment_data.get(variant, {}).get("SPR_BENCH", {}).get(epochs_str, {})


variants = ["baseline", "factorized"]
for var in variants:
    rec = get_run_record(var, "10")  # use the 10-epoch run for full curves
    if not rec:
        print(f"No record found for {var} 10 epochs")
        continue
    train_loss = rec["losses"]["train"]
    val_loss = rec["losses"]["val"]
    val_hwa = rec["metrics"]["val"]
    epochs = np.arange(1, len(train_loss) + 1)

    # -------- Loss curves --------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH {var.capitalize()} – Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{var}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {var}: {e}")
        plt.close()

    # -------- HWA curve --------
    try:
        plt.figure()
        plt.plot(epochs, val_hwa, marker="o", label="Val HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title(f"SPR_BENCH {var.capitalize()} – Validation HWA")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{var}_hwa.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot for {var}: {e}")
        plt.close()

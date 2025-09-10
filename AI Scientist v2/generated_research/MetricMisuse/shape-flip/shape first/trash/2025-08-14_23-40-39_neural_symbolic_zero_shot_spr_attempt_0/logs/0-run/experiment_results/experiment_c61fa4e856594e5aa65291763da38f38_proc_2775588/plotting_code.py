import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    sweep = experiment_data.get("weight_decay", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sweep = {}


# Helper to convert weight-decay string back to readable form for filenames
def sanitize(wd_str):
    return wd_str.replace(".", "p").replace("-", "m")


# 1-4: individual train/val loss curves
for wd_str, payload in list(sweep.items())[:4]:  # ensures max 4 similar figs
    try:
        metrics = payload["metrics"]
        epochs = np.arange(1, len(metrics["train_loss"]) + 1)
        plt.figure()
        plt.plot(epochs, metrics["train_loss"], label="Train Loss")
        plt.plot(epochs, metrics["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR Dataset\nTrain vs Val Loss (weight_decay={wd_str})")
        plt.legend()
        fname = f"spr_train_val_loss_wd_{sanitize(wd_str)}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for wd={wd_str}: {e}")
        plt.close()

# 5: Validation BPS comparison across weight decays
try:
    plt.figure()
    for wd_str, payload in sweep.items():
        epochs = np.arange(1, len(payload["metrics"]["val_bps"]) + 1)
        plt.plot(epochs, payload["metrics"]["val_bps"], label=f"wd={wd_str}")
    plt.xlabel("Epoch")
    plt.ylabel("BPS")
    plt.title("SPR Dataset\nValidation BPS Across Weight Decays")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_bps_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BPS comparison plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch series
def get_series(ds, key_chain):
    cur = experiment_data.get(ds, {})
    for k in key_chain:
        cur = cur.get(k, [])
    return cur


dataset_name = "SPR_BENCH"
train_hsca = get_series(dataset_name, ["metrics", "train"])
val_hsca = get_series(dataset_name, ["metrics", "val"])
train_loss = get_series(dataset_name, ["losses", "train"])

# -----------------------------------------------------------
# Plot 1: HSCA curves
try:
    if train_hsca:
        epochs = np.arange(1, len(train_hsca) + 1)
        plt.figure()
        plt.plot(epochs, train_hsca, marker="o", label="Train HSCA")
        if len(val_hsca) == len(train_hsca):
            plt.plot(epochs, val_hsca, marker="s", label="Validation/Test HSCA")
        plt.title(f"{dataset_name} – Harmonic SCA over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HSCA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dataset_name}_HSCA_curve.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("No HSCA data found; skipping HSCA plot.")
except Exception as e:
    print(f"Error creating HSCA plot: {e}")
    plt.close()

# -----------------------------------------------------------
# Plot 2: Training loss curve
try:
    if train_loss:
        epochs = np.arange(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, marker="x", color="tab:red", label="Train CE Loss")
        plt.title(f"{dataset_name} – Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dataset_name}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("No loss data found; skipping loss plot.")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -----------------------------------------------------------
# Print final evaluation metric if available
if val_hsca:
    print(f"Final Test HSCA: {val_hsca[-1]:.4f}")

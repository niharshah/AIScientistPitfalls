import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested keys
def safe_get(dic, keys, default=None):
    for k in keys:
        dic = dic.get(k, {})
    return dic if dic else default


# extract metrics
loss_train = safe_get(experiment_data, ["BiLSTM", "SPR_BENCH", "losses", "train"], [])
loss_val = safe_get(experiment_data, ["BiLSTM", "SPR_BENCH", "losses", "val"], [])
ccwa_val = safe_get(experiment_data, ["BiLSTM", "SPR_BENCH", "metrics", "val_CCWA"], [])

epochs = np.arange(1, len(loss_val) + 1)

# 1) train / val loss curve
try:
    plt.figure()
    if loss_train:
        plt.plot(epochs, loss_train, label="Train Loss")
    if loss_val:
        plt.plot(epochs, loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) validation CCWA curve
try:
    plt.figure()
    if ccwa_val:
        plt.plot(epochs, ccwa_val, marker="o", label="Validation CCWA")
    plt.xlabel("Epoch")
    plt.ylabel("CCWA Score")
    plt.title("SPR_BENCH: Validation CCWA over Epochs")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_CCWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CCWA curve: {e}")
    plt.close()

# print summary metrics
if ccwa_val:
    print(f"Best CCWA: {max(ccwa_val):.4f} at epoch {int(np.argmax(ccwa_val)+1)}")
    print(f"Final CCWA: {ccwa_val[-1]:.4f}")

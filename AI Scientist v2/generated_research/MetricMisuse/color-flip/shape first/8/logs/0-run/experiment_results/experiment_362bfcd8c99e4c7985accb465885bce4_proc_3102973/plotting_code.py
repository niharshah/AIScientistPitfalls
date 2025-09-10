import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_contrastive"]["SPR_BENCH"]
    epochs = list(range(1, len(exp["losses"]["train"]) + 1))
    train_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    val_ccwa = exp["metrics"]["val_CCWA"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    epochs, train_loss, val_loss, val_ccwa = [], [], [], []

# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss Curves (No Contrastive Pre-Training)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_no_contrastive_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2) Validation CCWA
try:
    plt.figure()
    plt.plot(epochs, val_ccwa, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("CCWA")
    plt.title("SPR_BENCH – Validation CCWA (No Contrastive Pre-Training)")
    fname = os.path.join(working_dir, "SPR_BENCH_no_contrastive_val_CCWA.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

print(f"Plots saved to: {working_dir}")

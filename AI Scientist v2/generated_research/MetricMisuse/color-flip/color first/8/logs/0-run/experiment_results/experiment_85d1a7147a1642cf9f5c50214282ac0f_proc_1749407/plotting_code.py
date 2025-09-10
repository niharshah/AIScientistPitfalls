import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["FrozenRandomEmbedding"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    # unpack losses
    train_loss = np.array(exp["losses"]["train"])  # shape (E,2) : epoch, loss
    val_loss = np.array(exp["losses"]["val"])  # shape (E,2)
    # unpack metrics (epoch, cwa, swa, hm, ocga)
    val_metrics = np.array(exp["metrics"]["val"])

    # ---------------- plot 1: loss curves ----------------
    try:
        plt.figure()
        plt.plot(train_loss[:, 0], train_loss[:, 1], label="Train")
        plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Loss Curves (FrozenRandomEmbedding)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve_FrozenRandomEmbedding.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------------- plot 2: validation metrics ----------------
    try:
        plt.figure()
        epochs = val_metrics[:, 0]
        labels = ["CWA", "SWA", "HM", "OCGA"]
        for i, lbl in enumerate(labels, start=1):
            plt.plot(epochs, val_metrics[:, i], label=lbl)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Validation Metrics (FrozenRandomEmbedding)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_metrics_curve_FrozenRandomEmbedding.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

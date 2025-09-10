import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- Synthetic1 ----------
try:
    s1 = experiment_data.get("Synthetic1", {})
    tr_loss, val_loss = s1["losses"]["train"], s1["losses"]["val"]
    if tr_loss and val_loss:
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Synthetic1: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "Synthetic1_loss_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Synthetic1 loss plot: {e}")
    plt.close()

try:
    cwca_tr, cwca_val = s1["metrics"]["train_cwca"], s1["metrics"]["val_cwca"]
    acr_tr, acr_val = s1["metrics"]["train_acr"], s1["metrics"]["val_acr"]
    if cwca_tr and acr_tr:
        epochs = np.arange(1, len(cwca_tr) + 1)
        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axs[0].plot(epochs, cwca_tr, label="Train")
        axs[0].plot(epochs, cwca_val, label="Validation")
        axs[0].set_ylabel("CWCA")
        axs[0].legend()
        axs[1].plot(epochs, acr_tr, label="Train")
        axs[1].plot(epochs, acr_val, label="Validation")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("ACR")
        axs[1].legend()
        fig.suptitle("Synthetic1: CWCA & ACR")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, "Synthetic1_metrics_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Synthetic1 metrics plot: {e}")
    plt.close()

# ---------- Synthetic2 ----------
try:
    s2 = experiment_data.get("Synthetic2", {})
    tr_loss, val_loss = s2["losses"]["train"], s2["losses"]["val"]
    if tr_loss and val_loss:
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Synthetic2: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "Synthetic2_loss_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Synthetic2 loss plot: {e}")
    plt.close()

try:
    cwca_tr, cwca_val = s2["metrics"]["train_cwca"], s2["metrics"]["val_cwca"]
    acr_tr, acr_val = s2["metrics"]["train_acr"], s2["metrics"]["val_acr"]
    if cwca_tr and acr_tr:
        epochs = np.arange(1, len(cwca_tr) + 1)
        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axs[0].plot(epochs, cwca_tr, label="Train")
        axs[0].plot(epochs, cwca_val, label="Validation")
        axs[0].set_ylabel("CWCA")
        axs[0].legend()
        axs[1].plot(epochs, acr_tr, label="Train")
        axs[1].plot(epochs, acr_val, label="Validation")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("ACR")
        axs[1].legend()
        fig.suptitle("Synthetic2: CWCA & ACR")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, "Synthetic2_metrics_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Synthetic2 metrics plot: {e}")
    plt.close()

# ---------- Synthetic3 ----------
try:
    s3 = experiment_data.get("Synthetic3", {})
    cwca_test = s3["metrics"]["test_cwca"]
    acr_test = s3["metrics"]["test_acr"]
    if cwca_test and acr_test:
        plt.figure()
        plt.bar(
            ["CWCA", "ACR"], [cwca_test[0], acr_test[0]], color=["skyblue", "salmon"]
        )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Synthetic3: Test Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "Synthetic3_test_metrics.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Synthetic3 metrics plot: {e}")
    plt.close()

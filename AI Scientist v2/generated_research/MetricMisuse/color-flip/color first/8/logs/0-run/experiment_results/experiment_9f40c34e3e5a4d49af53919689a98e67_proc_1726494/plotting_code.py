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


# Helper to extract arrays
def extract_series(hdim, key, idx):
    # key: 'losses' or 'metrics'; idx: column index within stored tuple
    data = experiment_data["hidden_dim"][hdim][key]["val" if key == "losses" else "val"]
    # for losses: list[(epoch, loss)], for metrics: list[(epoch,cwa,swa,cshm)]
    epochs = [t[0] for t in data]
    values = [t[idx] for t in data]
    return epochs, values


# 1) Train & Val loss curves
try:
    plt.figure()
    for hdim, rec in experiment_data.get("hidden_dim", {}).items():
        epochs_t, train_loss = zip(*rec["losses"]["train"])
        epochs_v, val_loss = zip(*rec["losses"]["val"])
        plt.plot(epochs_t, train_loss, label=f"h{hdim}-train")
        plt.plot(epochs_v, val_loss, linestyle="--", label=f"h{hdim}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Synthetic: Train vs Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_synthetic_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Validation CWA and SWA curves (two subplots)
try:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for hdim in experiment_data.get("hidden_dim", {}):
        epochs, cwa = extract_series(hdim, "metrics", 1)
        _, swa = extract_series(hdim, "metrics", 2)
        axs[0].plot(epochs, cwa, label=f"h{hdim}")
        axs[1].plot(epochs, swa, label=f"h{hdim}")
    axs[0].set_title("Left: CWA vs Epoch")
    axs[1].set_title("Right: SWA vs Epoch")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    fig.suptitle("SPR Synthetic Validation Accuracies")
    fname = os.path.join(working_dir, "spr_synthetic_cwa_swa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plots: {e}")
    plt.close()

# 3) Final CSHM comparison bar chart
try:
    plt.figure()
    hdims, finals = [], []
    for hdim in sorted(experiment_data.get("hidden_dim", {})):
        hdims.append(str(hdim))
        _, cshm = extract_series(hdim, "metrics", 3)
        finals.append(cshm[-1])  # last epoch value
    plt.bar(hdims, finals, color="skyblue")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Final CSHM")
    plt.title("SPR Synthetic: Final Harmonic Mean (CSHM) by Model Size")
    fname = os.path.join(working_dir, "spr_synthetic_final_cshm.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CSHM bar chart: {e}")
    plt.close()

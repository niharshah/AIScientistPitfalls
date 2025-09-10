import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick sanity check
sweep = experiment_data.get("dropout_tuning", {})
if not sweep:
    print("No dropout_tuning data found.")
    sweep = {}

# derive dataset name for figure titles
dataset_name = "SPR_BENCH" if os.getenv("SPR_PATH") else "Synthetic"

# gather keys
dropouts = sorted([float(k) for k in sweep.keys()])


# ------------------ helper to fetch arrays -------------------
def get_arr(k, subkey, d):
    return np.array(sweep[str(d)]["metrics" if k.endswith("MCC") else "losses"][subkey])


# ----------------- LOSS CURVES FIGURE ------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for d in dropouts:
        epochs = sweep[str(d)]["epochs"]
        train_loss = get_arr("loss", "train", d)
        val_loss = get_arr("loss", "val", d)
        axes[0].plot(epochs, train_loss, label=f"dropout={d}")
        axes[1].plot(epochs, val_loss, label=f"dropout={d}")
    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend()
    fig.suptitle(f"Loss Curves - {dataset_name}\nLeft: Train, Right: Validation")
    fig.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves figure: {e}")
    plt.close()

# ----------------- MCC CURVES FIGURE -------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for d in dropouts:
        epochs = sweep[str(d)]["epochs"]
        train_mcc = get_arr("MCC", "train_MCC", d)
        val_mcc = get_arr("MCC", "val_MCC", d)
        axes[0].plot(epochs, train_mcc, label=f"dropout={d}")
        axes[1].plot(epochs, val_mcc, label=f"dropout={d}")
    axes[0].set_title("Train MCC")
    axes[1].set_title("Validation MCC")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MCC")
        ax.legend()
    fig.suptitle(f"MCC Curves - {dataset_name}\nLeft: Train, Right: Validation")
    fig.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_mcc_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curves figure: {e}")
    plt.close()

# ----------------- TEST MCC BAR CHART ------------------------
try:
    test_mccs = [sweep[str(d)]["metrics"]["test_MCC"] for d in dropouts]
    plt.figure(figsize=(6, 4))
    plt.bar([str(d) for d in dropouts], test_mccs, color="skyblue")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Test MCC")
    plt.title(f"Test MCC by Dropout - {dataset_name}")
    for i, v in enumerate(test_mccs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fname = os.path.join(working_dir, f"{dataset_name}_test_mcc_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Test MCC bar chart: {e}")
    plt.close()

# ----------------- PRINT TEST MCC TABLE ----------------------
for d in dropouts:
    mcc = sweep[str(d)]["metrics"]["test_MCC"]
    print(f"dropout={d}: Test MCC={mcc:.4f}")

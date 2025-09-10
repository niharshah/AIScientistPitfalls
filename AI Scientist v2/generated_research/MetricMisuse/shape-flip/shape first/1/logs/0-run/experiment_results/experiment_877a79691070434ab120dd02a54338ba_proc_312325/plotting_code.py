import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["EPOCHS_TUNING"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# Helper: get ordered epoch values (e.g., 5,10,15,...)
epoch_settings = sorted(runs.keys())

# ---------------------------------------------------------------------
# 1) Validation accuracy curves
try:
    plt.figure()
    for ep in epoch_settings:
        vals = runs[ep]["metrics"]["val_acc"]
        plt.plot(range(1, len(vals) + 1), vals, label=f"{ep} epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR_BENCH: Validation Accuracy vs Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation accuracy plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) Training accuracy curves
try:
    plt.figure()
    for ep in epoch_settings:
        tr = runs[ep]["metrics"]["train_acc"]
        plt.plot(range(1, len(tr) + 1), tr, label=f"{ep} epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("SPR_BENCH: Training Accuracy vs Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Loss curves (train & val) in two subplots
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ep in epoch_settings:
        tl = runs[ep]["losses"]["train"]
        vl = runs[ep]["losses"]["val"]
        axes[0].plot(range(1, len(tl) + 1), tl, label=f"{ep} epochs")
        axes[1].plot(range(1, len(vl) + 1), vl, label=f"{ep} epochs")
    axes[0].set_title("Left: Train Loss")
    axes[1].set_title("Right: Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
    fig.suptitle("SPR_BENCH: Loss Curves Across Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 4) Final ZSRTA bar chart
try:
    zsrtas = [
        (
            np.nan
            if len(runs[ep]["metrics"]["ZSRTA"]) == 0
            else runs[ep]["metrics"]["ZSRTA"][-1]
        )
        for ep in epoch_settings
    ]
    plt.figure()
    plt.bar([str(ep) for ep in epoch_settings], zsrtas, color="skyblue")
    plt.xlabel("Epochs Trained")
    plt.ylabel("ZSRTA")
    plt.title("SPR_BENCH: Zero-Shot Rule Transfer Accuracy (ZSRTA)")
    fname = os.path.join(working_dir, "SPR_BENCH_ZSRTA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating ZSRTA bar plot: {e}")
    plt.close()

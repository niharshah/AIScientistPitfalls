import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load stored results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("no_length_masking", {}).get("SPR_BENCH", {})
hidden_dims = sorted(runs.keys(), key=lambda x: int(x.split("_")[-1]))


# helper to fetch metric list
def metric_per_run(metric_name):
    return [runs[h]["metrics"][metric_name] for h in hidden_dims]


# ------------------------------------------------------------------
# 1) Accuracy curves -------------------------------------------------
try:
    train_accs = metric_per_run("train_acc")
    val_accs = metric_per_run("val_acc")
    epochs = [range(1, len(a) + 1) for a in train_accs]

    plt.figure(figsize=(10, 4))
    plt.suptitle(
        "SPR_BENCH Accuracy Curves (No Length Masking)\n"
        "Left: Train Acc, Right: Validation Acc"
    )
    # left subplot: train
    plt.subplot(1, 2, 1)
    for ep, acc, hid in zip(epochs, train_accs, hidden_dims):
        plt.plot(ep, acc, label=hid)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train")
    plt.legend()

    # right subplot: val
    plt.subplot(1, 2, 2)
    for ep, acc, hid in zip(epochs, val_accs, hidden_dims):
        plt.plot(ep, acc, label=hid)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation")
    plt.legend()

    fname = os.path.join(working_dir, "spr_bench_accuracy_curves_no_length_masking.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Loss curves -----------------------------------------------------
try:
    train_losses = (
        metric_per_run("losses")["train"]
        if False
        else [runs[h]["losses"]["train"] for h in hidden_dims]
    )
    val_losses = [runs[h]["losses"]["val"] for h in hidden_dims]
    epochs = [range(1, len(l) + 1) for l in train_losses]

    plt.figure(figsize=(10, 4))
    plt.suptitle(
        "SPR_BENCH Loss Curves (No Length Masking)\n"
        "Left: Train Loss, Right: Validation Loss"
    )
    plt.subplot(1, 2, 1)
    for ep, ls, hid in zip(epochs, train_losses, hidden_dims):
        plt.plot(ep, ls, label=hid)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train")
    plt.legend()

    plt.subplot(1, 2, 2)
    for ep, ls, hid in zip(epochs, val_losses, hidden_dims):
        plt.plot(ep, ls, label=hid)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation")
    plt.legend()

    fname = os.path.join(working_dir, "spr_bench_loss_curves_no_length_masking.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) ZSRTA bar chart --------------------------------------------------
try:
    zsrta_vals = [
        runs[h]["metrics"]["ZSRTA"][0] if runs[h]["metrics"]["ZSRTA"] else np.nan
        for h in hidden_dims
    ]

    plt.figure(figsize=(6, 4))
    plt.title("SPR_BENCH Zero-Shot Rule-Transfer Accuracy (No Length Masking)")
    bars = plt.bar(hidden_dims, zsrta_vals, color="skyblue")
    plt.ylabel("ZSRTA")
    plt.ylim(0, 1)
    for b, v in zip(bars, zsrta_vals):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fname = os.path.join(working_dir, "spr_bench_zsrta_bars_no_length_masking.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating ZSRTA bar plot: {e}")
    plt.close()

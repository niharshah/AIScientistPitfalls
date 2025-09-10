import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------------
# gather lr keys and compute best lr via final val cpx
lr_dict = experiment_data.get("learning_rate", {})
best_lr_key, best_cpx = None, -float("inf")
for lr_key, lr_data in lr_dict.items():
    val_cpx_curve = [m["cpx"] for m in lr_data["SPR_BENCH"]["metrics"]["val"]]
    if val_cpx_curve and val_cpx_curve[-1] > best_cpx:
        best_cpx = val_cpx_curve[-1]
        best_lr_key = lr_key

# ---------------------------------------------------------------------
# PLOT 1: Validation CpxWA curves for all lrs
try:
    plt.figure()
    for lr_key, lr_data in lr_dict.items():
        epochs = lr_data["SPR_BENCH"]["epochs"]
        val_cpx = [m["cpx"] for m in lr_data["SPR_BENCH"]["metrics"]["val"]]
        plt.plot(epochs, val_cpx, marker="o", label=f"lr={lr_key}")
    plt.title(
        "SPR_BENCH – Validation Complexity-WA\nLeft: Multiple LRs, Right: Accuracy over Epochs"
    )
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_cpxwa_all_lrs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Validation CpxWA plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# PLOT 2: Training loss curves for all lrs
try:
    plt.figure()
    for lr_key, lr_data in lr_dict.items():
        epochs = lr_data["SPR_BENCH"]["epochs"]
        train_loss = lr_data["SPR_BENCH"]["losses"]["train"]
        plt.plot(epochs, train_loss, marker="o", label=f"lr={lr_key}")
    plt.title(
        "SPR_BENCH – Training Loss\nLeft: All Learning Rates, Right: Loss over Epochs"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_loss_all_lrs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Training Loss plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# PLOT 3: Train vs Val CpxWA for the best lr
if best_lr_key is not None:
    try:
        best_data = lr_dict[best_lr_key]["SPR_BENCH"]
        epochs = best_data["epochs"]
        val_cpx = [m["cpx"] for m in best_data["metrics"]["val"]]
        train_cpx = [m["cpx"] for m in best_data["metrics"]["train"]]
        plt.figure()
        plt.plot(epochs, train_cpx, marker="o", label="Train CpxWA")
        plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
        plt.title(f"SPR_BENCH – Train vs Val CpxWA (Best lr={best_lr_key})")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_bestlr_{best_lr_key}_train_vs_val_cpxwa.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Best-LR CpxWA plot: {e}")
        plt.close()

# ---------------------------------------------------------------------
# Print key metric
if best_lr_key is not None:
    print(f"Best learning rate: {best_lr_key}, final Val CpxWA: {best_cpx:.4f}")

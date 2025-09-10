import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

exp = experiment_data.get("epochs_tuning", {}).get("SPR_BENCH", None)
if exp is None:
    print("SPR_BENCH results not found in experiment_data.")
    raise SystemExit

epochs = np.array(exp["epochs"])
train_losses = np.array(exp["losses"]["train"])
val_metrics = exp["metrics"]["val"]
train_metrics = exp["metrics"]["train"]

val_cwa = np.array([m["cwa"] for m in val_metrics])
val_swa = np.array([m["swa"] for m in val_metrics])
val_cpx = np.array([m["cpx"] for m in val_metrics])
train_cpx = np.array([m["cpx"] for m in train_metrics])

best_epoch = int(epochs[np.argmax(val_cpx)])
best_val_cpx = float(val_cpx.max())

print(f"Best Validation CpxWA: {best_val_cpx:.4f} @ epoch {best_epoch}")

# ---------------------------------------------------------------------
# 1) Training loss curve
try:
    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.title("SPR_BENCH: Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating train loss plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) Training vs. Validation Complexity-Weighted Accuracy
try:
    plt.figure()
    plt.plot(epochs, train_cpx, marker="o", label="Train CpxWA")
    plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
    plt.title("SPR_BENCH: Complexity-Weighted Accuracy\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_cpxwa_train_val_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA curve: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Validation weighted-accuracy comparison
try:
    plt.figure()
    plt.plot(epochs, val_cwa, marker="o", label="Val CWA")
    plt.plot(epochs, val_swa, marker="^", label="Val SWA")
    plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
    plt.title("SPR_BENCH: Weighted Accuracy Comparison (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_weighted_accuracy_comparison.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy comparison plot: {e}")
    plt.close()

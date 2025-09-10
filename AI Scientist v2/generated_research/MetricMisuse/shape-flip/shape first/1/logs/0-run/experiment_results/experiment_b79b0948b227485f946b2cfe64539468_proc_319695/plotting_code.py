import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_runs = experiment_data.get("UniDirectional_GRU", {}).get("SPR_BENCH", {})
hidden_keys = sorted(
    spr_runs.keys(), key=lambda k: int(k.split("_")[-1])
)  # e.g. ['hidden_64', ...]

# quick containers for printing
summary_vals, summary_zs = [], []

# -------------------- Figure 1: Accuracy curves --------------------
try:
    plt.figure()
    for hk in hidden_keys:
        metrics = spr_runs[hk]["metrics"]
        epochs = range(1, len(metrics["train_acc"]) + 1)
        plt.plot(epochs, metrics["train_acc"], label=f"{hk}-train", linestyle="--")
        plt.plot(epochs, metrics["val_acc"], label=f"{hk}-val")
        summary_vals.append(metrics["val_acc"][-1])
        summary_zs.append(metrics["ZSRTA"][-1] if metrics["ZSRTA"] else np.nan)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves\nLeft: Training, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------------------- Figure 2: Loss curves --------------------
try:
    plt.figure()
    for hk in hidden_keys:
        losses = spr_runs[hk]["losses"]
        epochs = range(1, len(losses["train"]) + 1)
        plt.plot(epochs, losses["train"], label=f"{hk}-train", linestyle="--")
        plt.plot(epochs, losses["val"], label=f"{hk}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Training, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- Figure 3: ZSRTA bar chart --------------------
try:
    plt.figure()
    xs = np.arange(len(hidden_keys))
    plt.bar(xs, summary_zs, color="skyblue")
    plt.xticks(xs, hidden_keys)
    plt.ylabel("ZSRTA")
    plt.title("Zero-Shot Rule Transfer Accuracy (ZSRTA)\nDataset: SPR_BENCH")
    fname = os.path.join(working_dir, "SPR_BENCH_ZSRTA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating ZSRTA plot: {e}")
    plt.close()

# -------------------- Print summary table --------------------
print("HiddenDim\tFinalValAcc\tZSRTA")
for hk, v_acc, zs in zip(hidden_keys, summary_vals, summary_zs):
    print(f"{hk}\t{v_acc:.4f}\t{zs:.4f}")

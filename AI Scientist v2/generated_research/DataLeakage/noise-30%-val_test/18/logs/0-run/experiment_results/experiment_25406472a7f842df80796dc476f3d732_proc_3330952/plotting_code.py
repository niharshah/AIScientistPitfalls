import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("hidden_size", {}).get("SPR_BENCH", {})
if not runs:
    print("No SPR_BENCH data found, aborting plots.")
    exit()

# ---------------- helper ----------------
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
hs_keys = sorted(runs.keys(), key=lambda x: int(x.split("_")[1]))

# ------------- F1 curves ----------------
try:
    plt.figure()
    for idx, k in enumerate(hs_keys):
        epochs = runs[k]["epochs"]
        train_f1 = runs[k]["metrics"]["train"]
        val_f1 = runs[k]["metrics"]["val"]
        c = colors[idx % len(colors)]
        plt.plot(epochs, train_f1, linestyle="--", color=c, label=f"{k} train")
        plt.plot(epochs, val_f1, linestyle="-", color=c, label=f"{k} val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH F1 Curves\nLeft: Train (dashed), Right: Validation (solid)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ------------- Loss curves --------------
try:
    plt.figure()
    for idx, k in enumerate(hs_keys):
        epochs = runs[k]["epochs"]
        train_loss = runs[k]["losses"]["train"]
        val_loss = runs[k]["losses"]["val"]
        c = colors[idx % len(colors)]
        plt.plot(epochs, train_loss, linestyle="--", color=c, label=f"{k} train")
        plt.plot(epochs, val_loss, linestyle="-", color=c, label=f"{k} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train (dashed), Right: Validation (solid)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

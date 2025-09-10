import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

edge_data = experiment_data.get("edge_dropout", {})
tags = sorted(edge_data.keys())  # e.g. ['p_0.0', 'p_0.1', ...]

# ------------- plot 1: Loss curves -------------
try:
    plt.figure()
    for tag in tags:
        tr = edge_data[tag]["losses"]["train"]
        val = edge_data[tag]["losses"]["val"]
        epochs = np.arange(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{tag} train")
        plt.plot(epochs, val, linestyle="--", label=f"{tag} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss vs Epoch (all edge dropout rates)")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "spr_bench_loss_curves_all_dropout.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------- plot 2: Accuracy curves -------------
try:
    plt.figure()
    for tag in tags:
        tr = edge_data[tag]["metrics"]["train"]
        val = edge_data[tag]["metrics"]["val"]
        epochs = np.arange(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{tag} train")
        plt.plot(epochs, val, linestyle="--", label=f"{tag} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Accuracy vs Epoch (all edge dropout rates)")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "spr_bench_accuracy_curves_all_dropout.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ------------- plot 3: Complexity-Weighted Accuracy -------------
try:
    plt.figure()
    cwas = [edge_data[tag].get("comp_weighted_acc", 0.0) for tag in tags]
    plt.bar(tags, cwas, color="skyblue")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH – Final CWA vs Edge Dropout")
    plt.xticks(rotation=45)
    fname = os.path.join(working_dir, "spr_bench_cwa_vs_dropout.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating CWA bar plot: {e}")
    plt.close()

print("Saved figures:", [f for f in os.listdir(working_dir) if f.endswith(".png")])

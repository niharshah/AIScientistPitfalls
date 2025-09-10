import matplotlib.pyplot as plt
import numpy as np
import os

# setup paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = {}

epochs = len(spr.get("losses", {}).get("train", []))

# ---------- Plot 1: Train / Val loss curves ----------
try:
    if epochs:
        x = np.arange(1, epochs + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(x, spr["losses"]["train"], label="train", linestyle="--")
        plt.plot(x, spr["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No loss data to plot.")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
finally:
    plt.close()

# ---------- Plot 2: Weighted accuracy curves ----------
try:
    if epochs:
        x = np.arange(1, epochs + 1)
        plt.figure(figsize=(6, 4))
        for met in ["SWA", "CWA", "CoWA"]:
            plt.plot(x, spr["metrics"][met], label=met)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Weighted Accuracy Metrics")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_weighted_acc.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No metric data to plot.")
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
finally:
    plt.close()

# ---------- Plot 3: Plain accuracy per epoch ----------
try:
    if epochs:
        acc = []
        for p, g in zip(spr["predictions"], spr["ground_truth"]):
            p_arr, g_arr = np.array(p), np.array(g)
            acc.append(float(np.mean(p_arr == g_arr)))
        x = np.arange(1, epochs + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(x, acc, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Plain Validation Accuracy")
        fname = os.path.join(working_dir, "SPR_BENCH_plain_accuracy.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No predictions to compute accuracy.")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
finally:
    plt.close()

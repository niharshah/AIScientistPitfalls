import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data ------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    # helpers
    def get_runs(ds_name="SPR_BENCH"):
        runs = experiment_data.get("learning_rate", {}).get(ds_name, {})
        return {float(k): v for k, v in runs.items()}

    runs = get_runs()
    epochs = range(1, 1 + max(len(v["losses"]["train"]) for v in runs.values()))

    colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))  # distinct colors

    # ------------ plot losses ------------
    try:
        plt.figure(figsize=(7, 4))
        for (lr, run), c in zip(runs.items(), colors):
            plt.plot(
                epochs, run["losses"]["train"], "--", color=c, label=f"Train lr={lr}"
            )
            plt.plot(epochs, run["losses"]["val"], "-", color=c, label=f"Val lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Loss Curves\nTrain vs Validation across learning rates")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------ plot accuracy ------------
    try:
        plt.figure(figsize=(7, 4))
        for (lr, run), c in zip(runs.items(), colors):
            acc = [m["acc"] for m in run["metrics"]["val"]]
            plt.plot(epochs, acc, "-o", color=c, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("SPR_BENCH – Validation Accuracy across learning rates")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------ plot PCWA ------------
    try:
        plt.figure(figsize=(7, 4))
        for (lr, run), c in zip(runs.items(), colors):
            pcwa = [m["pcwa"] for m in run["metrics"]["val"]]
            plt.plot(epochs, pcwa, "-s", color=c, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("PC-Weighted Accuracy")
        plt.title("SPR_BENCH – PC-Weighted Accuracy across learning rates")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_pcwa.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating PCWA plot: {e}")
        plt.close()

    # ------------ print numerical summary ------------
    print("\nBest Validation Accuracy by Learning Rate")
    for lr, run in sorted(runs.items()):
        best_acc = max(m["acc"] for m in run["metrics"]["val"])
        print(f"  lr={lr:<6}: {best_acc:.3f}")

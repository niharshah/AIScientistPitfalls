import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data["learning_rate"]["SPR"]
    lrs = sorted(runs.keys(), key=lambda x: float(x.replace("e-", "e-0")))
    epochs = len(next(iter(runs.values()))["losses"]["train"])

    # Helper to pull metric list
    def metric_list(run, split, field):
        if field == "losses":
            return run["losses"][split]
        else:  # metrics
            return [m[field] for m in run["metrics"][split]]

    # ----------- Figure 1: Train/Val Loss -----------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        for lr in lrs:
            run = runs[lr]
            ax[0].plot(
                range(1, epochs + 1),
                metric_list(run, "train", "losses"),
                label=f"lr={lr}",
            )
            ax[1].plot(
                range(1, epochs + 1),
                metric_list(run, "val", "losses"),
                label=f"lr={lr}",
            )
        ax[0].set_title("Left: Train Loss")
        ax[1].set_title("Right: Validation Loss")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Loss")
            a.legend()
        fig.suptitle("SPR Loss Curves Across Learning Rates")
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ----------- Figure 2: Accuracy & CAA -----------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        for lr in lrs:
            run = runs[lr]
            acc = [m["acc"] for m in run["metrics"]["val"]]
            caa = [m["caa"] for m in run["metrics"]["val"]]
            ax[0].plot(range(1, epochs + 1), acc, label=f"lr={lr}")
            ax[1].plot(range(1, epochs + 1), caa, label=f"lr={lr}")
        ax[0].set_title("Left: Accuracy")
        ax[1].set_title("Right: Complexity Adjusted Accuracy")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Score")
            a.legend()
        fig.suptitle("SPR Basic vs Complexity-Aware Accuracy")
        fname = os.path.join(working_dir, "SPR_accuracy_vs_caa.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating accuracy/CAA plot: {e}")
        plt.close()

    # ----------- Figure 3: Color vs Shape Weighted Acc -----------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        for lr in lrs:
            run = runs[lr]
            cwa = [m["cwa"] for m in run["metrics"]["val"]]
            swa = [m["swa"] for m in run["metrics"]["val"]]
            ax[0].plot(range(1, epochs + 1), cwa, label=f"lr={lr}")
            ax[1].plot(range(1, epochs + 1), swa, label=f"lr={lr}")
        ax[0].set_title("Left: Color-Weighted Accuracy")
        ax[1].set_title("Right: Shape-Weighted Accuracy")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Score")
            a.legend()
        fig.suptitle("SPR Color vs Shape Weighted Accuracies")
        fname = os.path.join(working_dir, "SPR_cwa_swa_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating CWA/SWA plot: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------- prepare -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR" in experiment_data:
    run = experiment_data["SPR"]  # single run
    epochs = len(run["losses"]["train"])

    # helper to fetch metric list safely
    def get_metric(lst, key):
        return [m.get(key, np.nan) for m in lst] if lst else [np.nan] * epochs

    # ============== Figure 1: Loss curves ==============
    try:
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=120)
        ax[0].plot(range(1, epochs + 1), run["losses"]["train"], label="Train")
        ax[1].plot(
            range(1, epochs + 1), run["losses"]["val"], label="Val", color="orange"
        )
        ax[0].set_title("Train Loss")
        ax[1].set_title("Validation Loss")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Loss")
            a.legend()
        fig.suptitle("SPR Dataset — Left: Train Loss, Right: Validation Loss")
        fname = os.path.join(working_dir, "SPR_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ============== Figure 2: Validation metrics ==============
    try:
        acc = get_metric(run["metrics"]["val"], "acc")
        cwa = get_metric(run["metrics"]["val"], "cwa")
        swa = get_metric(run["metrics"]["val"], "swa")
        hpa = get_metric(run["metrics"]["val"], "hpa")

        fig, ax = plt.subplots(2, 2, figsize=(10, 6), dpi=120)
        titles = [
            "Accuracy",
            "Color-Weighted Accuracy (CWA)",
            "Shape-Weighted Accuracy (SWA)",
            "Harmonic (HPA)",
        ]
        data = [acc, cwa, swa, hpa]
        for i, a in enumerate(ax.flatten()):
            a.plot(range(1, epochs + 1), data[i])
            a.set_title(titles[i])
            a.set_xlabel("Epoch")
            a.set_ylabel("Score")
            a.set_ylim(0, 1)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("SPR Dataset — Validation Metrics per Epoch")
        fname = os.path.join(working_dir, "SPR_val_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()
else:
    print("No SPR data found in experiment_data.npy")

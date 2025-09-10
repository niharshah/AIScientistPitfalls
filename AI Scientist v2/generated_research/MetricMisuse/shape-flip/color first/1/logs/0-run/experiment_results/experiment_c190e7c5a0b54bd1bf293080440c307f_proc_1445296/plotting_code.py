import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR" in experiment_data:
    spr = experiment_data["SPR"]
    ep = spr.get("epochs", list(range(1, len(spr["losses"]["train"]) + 1)))

    # ---- helper lambdas ----
    def arr(path, default=[]):
        cur = spr
        for k in path.split("/"):
            cur = cur.get(k, {})
        return cur if isinstance(cur, (list, np.ndarray)) else default

    # --------------- Figure 1: Loss curves ---------------
    try:
        plt.figure(figsize=(6, 4), dpi=120)
        plt.plot(ep, arr("losses/train"), label="Train")
        plt.plot(ep, arr("losses/val"), label="Validation")
        plt.title("SPR Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------------- Figure 2: Validation metrics ---------------
    try:
        metrics = spr["metrics"]["val"]
        keys = ["acc", "CWA", "SWA", "HPA"]
        figs, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=120)
        ax = ax.flatten()
        for i, k in enumerate(keys):
            ax[i].plot(ep, [m[k] for m in metrics], label=k)
            ax[i].set_title(k)
            ax[i].set_xlabel("Epoch")
            ax[i].set_ylabel("Score")
            ax[i].set_ylim(0, 1)
        plt.suptitle("SPR Validation Metrics Over Epochs")
        fname = os.path.join(working_dir, "SPR_validation_metrics.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # --------------- Figure 3: Test prediction distribution ---------------
    try:
        preds = np.array(spr.get("predictions", []))
        gts = np.array(spr.get("ground_truth", []))
        if preds.size and gts.size:
            classes = sorted(set(gts) | set(preds))
            width = 0.35
            x = np.arange(len(classes))
            plt.figure(figsize=(6, 4), dpi=120)
            plt.bar(
                x - width / 2,
                [np.sum(gts == c) for c in classes],
                width=width,
                label="Ground Truth",
            )
            plt.bar(
                x + width / 2,
                [np.sum(preds == c) for c in classes],
                width=width,
                label="Predictions",
            )
            plt.xticks(x, classes)
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title("SPR Test Set: Ground Truth vs Predictions")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_test_distribution.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()
else:
    print("No SPR data available to plot.")

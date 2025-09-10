import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

joint = experiment_data.get("joint_train", {})
losses = joint.get("losses", {})
metrics = joint.get("metrics", {})
preds = joint.get("predictions", [])
gts = joint.get("ground_truth", [])


# helper
def _epochs(x):  # returns 1..len(x)
    return list(range(1, len(x) + 1))


# -------------- FIGURE 1: Loss curves -------------------
try:
    tr_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if tr_loss and val_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(_epochs(tr_loss), tr_loss, label="Train")
        plt.plot(_epochs(val_loss), val_loss, label="Validation")
        plt.title("SPR_BENCH: Train vs Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# -------------- FIGURE 2: Metric curves -----------------
try:
    val_metrics = metrics.get("val", [])
    if val_metrics:
        epochs = [m["epoch"] for m in val_metrics]
        for key in ("swa", "cwa", "ccwa"):
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, [m[key] for m in val_metrics], marker="o")
            plt.title(f"SPR_BENCH: Validation {key.upper()} Across Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(key.upper())
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"SPR_BENCH_{key}_curve.png"))
            plt.close()
    else:
        plt.close()
except Exception as e:
    print(f"Error creating metric curves plot: {e}")
    plt.close()

# -------------- FIGURE 3: Best metric summary -----------
try:
    if val_metrics:
        best = max(val_metrics, key=lambda x: x["ccwa"])
        labels = ["SWA", "CWA", "CCWA"]
        vals = [best["swa"], best["cwa"], best["ccwa"]]
        plt.figure(figsize=(5, 4))
        plt.bar(labels, vals)
        plt.title("SPR_BENCH: Best-Epoch Metrics")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_best_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating best-metric bar plot: {e}")
    plt.close()

# -------------- FIGURE 4: Confusion matrix --------------
try:
    if preds and gts:
        import itertools
        import numpy as np

        labels = sorted(set(gts + preds))
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title("SPR_BENCH: Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(labels)
        plt.yticks(labels)
        for i, j in itertools.product(range(n), range(n)):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=6
            )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

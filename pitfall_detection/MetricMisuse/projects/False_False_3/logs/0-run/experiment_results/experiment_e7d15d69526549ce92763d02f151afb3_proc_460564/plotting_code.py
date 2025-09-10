import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# --------------------------------------------------------------------- paths / data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds = "SPR_BENCH"
if ds not in experiment_data:
    print(f"{ds} data not found in experiment_data.npy")
    exit()

data = experiment_data[ds]


# --------------------------------------------------------------------- helper
def safe_len(lst):
    return 0 if lst is None else len(lst)


# --------------------------------------------------------------------- Plot 1: Loss curves
try:
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    epochs = range(1, len(val_losses) + 1)

    plt.figure()
    if safe_len(train_losses):
        plt.plot(epochs, train_losses[: len(epochs)], label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ds} – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# --------------------------------------------------------------------- Plot 2: Validation metrics
try:
    val_metrics = np.array([m for m in data["metrics"]["val"] if m is not None])
    if val_metrics.size:
        swa, cwa, rcaa = val_metrics.T
        epochs = range(1, len(swa) + 1)
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, rcaa, label="RCAA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds} – Validation Metrics Over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_val_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    else:
        print("No validation metrics found to plot.")
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# --------------------------------------------------------------------- Plot 3: Label distribution (true vs predicted)
try:
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])
    if preds and gts:
        labels = sorted(set(gts + preds))
        pred_cnt = [preds.count(l) for l in labels]
        true_cnt = [gts.count(l) for l in labels]

        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(max(6, len(labels)), 4))
        plt.bar(x - width / 2, true_cnt, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_cnt, width, label="Predicted")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title(f"{ds} – Label Distribution (True vs Predicted)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds}_label_distribution.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    else:
        print("Predictions or ground truth missing; skipping distribution plot.")
except Exception as e:
    print(f"Error creating distribution plot: {e}")
    plt.close()

# --------------------------------------------------------------------- Evaluation prints
if preds and gts:
    overall_acc = sum(p == t for p, t in zip(preds, gts)) / len(gts)
    print(f"Overall accuracy on stored split: {overall_acc:.3f}")

    # If metrics were stored per sample, we could recompute; here we reuse last epoch val metrics if available
    if val_metrics.size:
        print(
            f"Last recorded validation SWA={swa[-1]:.3f}, CWA={cwa[-1]:.3f}, RCAA={rcaa[-1]:.3f}"
        )

import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# basic set-up
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("SPR", {})
epochs = ed.get("epochs", [])
train_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
train_met = ed.get("metrics", {}).get("train", [])
val_met = ed.get("metrics", {}).get("val", [])
preds = np.asarray(ed.get("predictions", []), dtype=int)
gts = np.asarray(ed.get("ground_truth", []), dtype=int)
best_ep = ed.get("best_epoch", None)

# ------------------------------------------------------------------ #
# 1) loss curves
# ------------------------------------------------------------------ #
try:
    if epochs and train_loss and val_loss:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR – Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) metric curves (CWA, SWA, HWA)
# ------------------------------------------------------------------ #
try:
    if train_met and val_met:
        plt.figure()
        for key, lab in [("CWA", "CWA"), ("SWA", "SWA"), ("HWA", "HWA")]:
            plt.plot(epochs, [m[key] for m in train_met], label=f"Train {lab}")
            plt.plot(
                epochs, [m[key] for m in val_met], linestyle="--", label=f"Val {lab}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR – Weighted Metrics Over Epochs")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_metric_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) confusion matrix
# ------------------------------------------------------------------ #
try:
    if preds.size and gts.size:
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR – Confusion Matrix (Test Set)")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=7,
                )
        fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
        plt.savefig(fname, dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) per-class accuracy bar chart
# ------------------------------------------------------------------ #
try:
    if preds.size and gts.size:
        n_cls = int(max(preds.max(), gts.max()) + 1)
        acc = [
            (preds[gts == i] == i).mean() if (gts == i).sum() else 0
            for i in range(n_cls)
        ]
        plt.figure()
        plt.bar(range(n_cls), acc)
        plt.xlabel("Class ID")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title("SPR – Per-Class Test Accuracy")
        fname = os.path.join(working_dir, "SPR_per_class_accuracy.png")
        plt.savefig(fname, dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating per-class accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Print summary metrics
# ------------------------------------------------------------------ #
if val_met:
    best_idx = best_ep - 1 if best_ep is not None else -1
    best_metrics = val_met[best_idx] if 0 <= best_idx < len(val_met) else val_met[-1]
    print(f"Best epoch: {best_ep}")
    print(
        f"Test CWA: {best_metrics.get('CWA', float('nan')):.3f}, "
        f"SWA: {best_metrics.get('SWA', float('nan')):.3f}, "
        f"HWA: {best_metrics.get('HWA', float('nan')):.3f}"
    )

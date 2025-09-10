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

# ------------------------------------------------------------------ #
# helper access
# ------------------------------------------------------------------ #
ed = experiment_data.get("SPR", {})
lr_vals = ed.get("lr_values", [])
epochs = ed.get("epochs", [])
best_lr = ed.get("best_lr", None)
best_idx = lr_vals.index(best_lr) if best_lr in lr_vals else -1

# ------------------------------------------------------------------ #
# 1) loss curves (train/val) for best LR
# ------------------------------------------------------------------ #
try:
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"][best_idx], label="Train")
    plt.plot(epochs, ed["losses"]["val"][best_idx], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"SPR – Train vs Val Loss (best lr={best_lr})")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_loss_curves_lr{best_lr}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) HWA metric curves (train/val) for best LR
# ------------------------------------------------------------------ #
try:
    tr_hwa = [m["HWA"] for m in ed["metrics"]["train"][best_idx]]
    val_hwa = [m["HWA"] for m in ed["metrics"]["val"][best_idx]]
    plt.figure()
    plt.plot(epochs, tr_hwa, label="Train HWA")
    plt.plot(epochs, val_hwa, label="Val HWA")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title(f"SPR – HWA Curves (best lr={best_lr})")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_HWA_curves_lr{best_lr}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) bar chart of final Val-HWA vs LR
# ------------------------------------------------------------------ #
try:
    final_hwa = [metrics[-1]["HWA"] for metrics in ed["metrics"]["val"]]
    plt.figure()
    plt.bar([str(lr) for lr in lr_vals], final_hwa)
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Val HWA")
    plt.title("SPR – Final Validation HWA per Learning Rate")
    fname = os.path.join(working_dir, "SPR_valHWA_vs_lr.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating LR sweep HWA bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) confusion matrix on test split
# ------------------------------------------------------------------ #
try:
    preds = np.asarray(ed.get("predictions", []), dtype=int)
    gts = np.asarray(ed.get("ground_truth", []), dtype=int)
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
                    fontsize=8,
                )
        fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
        plt.savefig(fname, dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# print key metrics
# ------------------------------------------------------------------ #
try:
    last_val_metrics = ed["metrics"]["val"][best_idx][-1]
    print(f"Best LR: {best_lr}")
    print(
        f"Final Val Metrics – CWA: {last_val_metrics['CWA']:.4f} | "
        f"SWA: {last_val_metrics['SWA']:.4f} | HWA: {last_val_metrics['HWA']:.4f}"
    )
except Exception as e:
    print(f"Error printing metrics: {e}")

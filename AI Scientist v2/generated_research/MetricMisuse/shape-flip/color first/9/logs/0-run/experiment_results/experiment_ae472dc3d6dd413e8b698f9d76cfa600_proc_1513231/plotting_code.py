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
# helper variables
# ------------------------------------------------------------------ #
ed = experiment_data.get("SPR", {})
lr_vals = ed.get("lr_values", [])
epochs = ed.get("epochs", [])
best_lr = ed.get("best_lr", None)
best_idx = lr_vals.index(best_lr) if best_lr in lr_vals else -1

# ------------------------------------------------------------------ #
# 1) loss curves (train / val) for best LR
# ------------------------------------------------------------------ #
try:
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"][best_idx], label="Train")
    plt.plot(epochs, ed["losses"]["val"][best_idx], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"SPR Dataset – Train/Val Loss (best lr={best_lr})")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_loss_curves_lr{best_lr}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) HM curves (train / val) for best LR
# ------------------------------------------------------------------ #
try:
    tr_hm = [m["HM"] for m in ed["metrics"]["train"][best_idx]]
    val_hm = [m["HM"] for m in ed["metrics"]["val"][best_idx]]
    plt.figure()
    plt.plot(epochs, tr_hm, label="Train HM")
    plt.plot(epochs, val_hm, label="Val HM")
    plt.xlabel("Epoch")
    plt.ylabel("HM")
    plt.title(f"SPR Dataset – Harmonic-Mean Metric (best lr={best_lr})")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_HM_curves_lr{best_lr}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HM curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) bar chart of final Val-HM vs LR
# ------------------------------------------------------------------ #
try:
    final_hms = [metrics[-1]["HM"] for metrics in ed["metrics"]["val"]]
    plt.figure()
    plt.bar([str(lr) for lr in lr_vals], final_hms)
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Val HM")
    plt.title("SPR Dataset – Final Validation HM per Learning Rate")
    fname = os.path.join(working_dir, "SPR_valHM_vs_lr.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating LR sweep HM bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) confusion matrix on test split
# ------------------------------------------------------------------ #
try:
    preds = np.asarray(ed.get("predictions", []), dtype=int)
    gts = np.asarray(ed.get("ground_truth", []), dtype=int)
    if preds.size and gts.size:
        n_cls = max(preds.max(), gts.max()) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR Dataset – Confusion Matrix (Test Set)")
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
# print stored test metrics
# ------------------------------------------------------------------ #
tm = ed.get("metrics", {}).get("val", [{}])[-1] if ed else {}
print(f"Best LR: {best_lr}")
print(
    f"Test metrics – CWA: {tm.get('CWA', 'NA'):.4f} | "
    f"SWA: {tm.get('SWA', 'NA'):.4f} | HM: {tm.get('HM', 'NA'):.4f}"
)

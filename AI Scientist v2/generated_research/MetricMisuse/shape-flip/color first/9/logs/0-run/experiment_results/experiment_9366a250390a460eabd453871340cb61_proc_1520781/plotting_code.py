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
loss_tr = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])
hwa_tr = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("train", [])]
hwa_val = [m.get("HWA", np.nan) for m in ed.get("metrics", {}).get("val", [])]
preds = np.asarray(ed.get("predictions", []), dtype=int)
gts = np.asarray(ed.get("ground_truth", []), dtype=int)

# ------------------------------------------------------------------ #
# 1) Train / Val loss curve
# ------------------------------------------------------------------ #
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR – Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) Train / Val HWA curve
# ------------------------------------------------------------------ #
try:
    plt.figure()
    plt.plot(epochs, hwa_tr, label="Train HWA")
    plt.plot(epochs, hwa_val, label="Val HWA")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR – Train vs Val HWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_HWA_curve.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) Confusion matrix on test set
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
        plt.title("SPR – Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
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
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"), dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) Final CWA / SWA / HWA bar chart
# ------------------------------------------------------------------ #
try:
    if hwa_val:
        cwa_last = ed.get("metrics", {}).get("val", [{}])[-1].get("CWA", np.nan)
        swa_last = ed.get("metrics", {}).get("val", [{}])[-1].get("SWA", np.nan)
        hwa_last = ed.get("metrics", {}).get("val", [{}])[-1].get("HWA", np.nan)
        cats = ["CWA", "SWA", "HWA"]
        vals = [cwa_last, swa_last, hwa_last]
        plt.figure()
        plt.bar(cats, vals, color=["#4c72b0", "#55a868", "#c44e52"])
        plt.ylabel("Metric Value")
        plt.title("SPR – Final Validation Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_final_val_metrics.png"), dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating metric bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Print simple evaluation metric
# ------------------------------------------------------------------ #
if preds.size and gts.size:
    accuracy = (preds == gts).mean()
    print(f"Test accuracy: {accuracy:.4f}")

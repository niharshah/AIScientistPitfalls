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

# ------------------------------------------------------------------ #
# 1) loss curves
# ------------------------------------------------------------------ #
try:
    if train_loss and val_loss:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset – Train vs Val Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_train_val_loss.png")
        plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) HWA curves
# ------------------------------------------------------------------ #
try:
    if train_met and val_met:
        tr_hwa = [m["HWA"] for m in train_met]
        val_hwa = [m["HWA"] for m in val_met]
        plt.figure()
        plt.plot(epochs, tr_hwa, label="Train HWA")
        plt.plot(epochs, val_hwa, label="Val HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR Dataset – Train vs Val HWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_train_val_HWA.png")
        plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) bar chart of final Val-HWA
# ------------------------------------------------------------------ #
try:
    if val_met:
        final_hwa = val_met[-1]["HWA"]
        plt.figure()
        plt.bar(["run_1"], [final_hwa])
        plt.ylabel("Final Val HWA")
        plt.title("SPR Dataset – Final Validation HWA")
        fname = os.path.join(working_dir, "SPR_final_val_HWA.png")
        plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating final Val HWA bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) confusion matrix
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
        plt.title("SPR Dataset – Confusion Matrix (Test)")
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
# Print simple test accuracy
# ------------------------------------------------------------------ #
if preds.size and gts.size:
    acc = (preds == gts).mean()
    print(f"Test accuracy: {acc:.4f}")

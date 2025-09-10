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
# helper
# ------------------------------------------------------------------ #
def slice_by_lr(lst, lr_idx, epochs):
    """Return sub-list that corresponds to the lr_idx-th learning rate."""
    start = lr_idx * epochs
    end = start + epochs
    return lst[start:end]


# ------------------------------------------------------------------ #
# iterate over all datasets in experiment_data
# ------------------------------------------------------------------ #
for dname, ed in experiment_data.items():
    lr_vals = ed.get("lr_values", [])
    epochs = ed.get("epochs", [])
    best_lr = ed.get("best_lr", None)
    if not lr_vals or not epochs:
        print(f"Dataset {dname} missing lr/epoch information – skipping.")
        continue

    ep_cnt = len(epochs)
    try:
        best_idx = lr_vals.index(best_lr)
    except ValueError:
        best_idx = -1

    # ------------------------------------------------------------------ #
    # loss curves (train / val) for best LR
    # ------------------------------------------------------------------ #
    try:
        tr_losses = slice_by_lr(ed["losses"]["train"], best_idx, ep_cnt)
        val_losses = slice_by_lr(ed["losses"]["val"], best_idx, ep_cnt)
        plt.figure()
        plt.plot(epochs, tr_losses, label="Train")
        plt.plot(epochs, val_losses, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname} – Loss Curves (best lr={best_lr})")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curves_bestlr.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # HWA curves (train / val) for best LR
    # ------------------------------------------------------------------ #
    try:
        tr_mets = slice_by_lr(ed["metrics"]["train"], best_idx, ep_cnt)
        val_mets = slice_by_lr(ed["metrics"]["val"], best_idx, ep_cnt)
        tr_hwa = [m["HWA"] for m in tr_mets]
        val_hwa = [m["HWA"] for m in val_mets]
        plt.figure()
        plt.plot(epochs, tr_hwa, label="Train HWA")
        plt.plot(epochs, val_hwa, label="Val HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title(f"{dname} – HWA Curves (best lr={best_lr})")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_HWA_curves_bestlr.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting HWA curves for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Bar chart of final Val-HWA vs LR
    # ------------------------------------------------------------------ #
    try:
        final_val_hwa = [
            ed["metrics"]["val"][i * ep_cnt + (ep_cnt - 1)]["HWA"]
            for i in range(len(lr_vals))
        ]
        plt.figure()
        plt.bar([str(lr) for lr in lr_vals], final_val_hwa)
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Val HWA")
        plt.title(f"{dname} – Final Validation HWA per LR")
        fname = os.path.join(working_dir, f"{dname}_valHWA_vs_lr.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting LR sweep for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Confusion matrix on test split
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
            plt.title(f"{dname} – Confusion Matrix (Test Set)")
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
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Print numeric summary for best LR
    # ------------------------------------------------------------------ #
    try:
        best_val_hwa = val_hwa[-1] if val_hwa else "NA"
        print(f"[{dname}] Best LR: {best_lr} – Final Val HWA: {best_val_hwa:.4f}")
    except Exception:
        pass

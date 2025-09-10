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
# iterate through each dataset block
# ------------------------------------------------------------------ #
for dname, dct in experiment_data.items():
    epochs = dct.get("epochs", [])
    tr_loss = dct.get("losses", {}).get("train", [])
    val_loss = dct.get("losses", {}).get("val", [])
    tr_met = dct.get("metrics", {}).get("train", [])
    val_met = dct.get("metrics", {}).get("val", [])
    preds = np.asarray(dct.get("predictions", []), dtype=int)
    gts = np.asarray(dct.get("ground_truth", []), dtype=int)

    # ---------------------------#
    # 1) loss curves
    # ---------------------------#
    try:
        if epochs and tr_loss and val_loss:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname} – Train vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # ---------------------------#
    # 2) HWA metric curves
    # ---------------------------#
    try:
        if epochs and tr_met and val_met:
            tr_hwa = [m.get("HWA", np.nan) for m in tr_met]
            val_hwa = [m.get("HWA", np.nan) for m in val_met]
            plt.figure()
            plt.plot(epochs, tr_hwa, label="Train HWA")
            plt.plot(epochs, val_hwa, label="Validation HWA")
            plt.xlabel("Epoch")
            plt.ylabel("HWA")
            plt.title(f"{dname} – Train vs Validation HWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_HWA_curves.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating HWA plot for {dname}: {e}")
        plt.close()

    # ---------------------------#
    # 3) confusion matrix
    # ---------------------------#
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
            plt.title(f"{dname} – Confusion Matrix (Test)")
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
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # ---------------------------#
    # print final metrics
    # ---------------------------#
    try:
        if val_met:
            last_val = val_met[-1]
            print(
                f"{dname} – Final Validation metrics:"
                f" CWA={last_val.get('CWA', 'NA'):.4f},"
                f" SWA={last_val.get('SWA', 'NA'):.4f},"
                f" HWA={last_val.get('HWA', 'NA'):.4f}"
            )
    except Exception as e:
        print(f"Error printing metrics for {dname}: {e}")

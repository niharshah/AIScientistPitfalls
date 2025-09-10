import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset, rec in experiment_data.items():
    epochs = np.array(rec.get("epochs", []))
    train_l = np.array(rec.get("losses", {}).get("train", []), dtype=float)
    val_l = np.array(rec.get("losses", {}).get("val", []), dtype=float)
    swa = np.array(
        [m["SWA"] if m else np.nan for m in rec.get("metrics", {}).get("val", [])],
        dtype=float,
    )
    nwa = np.array(
        [m["NWA"] if m else np.nan for m in rec.get("metrics", {}).get("val", [])],
        dtype=float,
    )
    preds = rec.get("predictions", [])[-1] if rec.get("predictions") else None
    gts = rec.get("ground_truth", [])[-1] if rec.get("ground_truth") else None

    # -------- 1. loss curves -------------------------------------------------
    try:
        if len(epochs):
            plt.figure()
            if len(train_l):
                plt.plot(epochs, train_l, label="Train Loss")
            if len(val_l):
                plt.plot(epochs, val_l, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset} – Training vs Validation Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset}: {e}")
        plt.close()

    # -------- 2. SWA curve ---------------------------------------------------
    try:
        if len(epochs) and not np.isnan(swa).all():
            plt.figure()
            plt.plot(epochs, swa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{dset} – Validation SWA over Epochs")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_SWA_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error plotting SWA for {dset}: {e}")
        plt.close()

    # -------- 3. NWA curve ---------------------------------------------------
    try:
        if len(epochs) and not np.isnan(nwa).all():
            plt.figure()
            plt.plot(epochs, nwa, marker="o", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Novelty-Weighted Accuracy")
            plt.title(f"{dset} – Validation NWA over Epochs")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_NWA_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error plotting NWA for {dset}: {e}")
        plt.close()

    # -------- 4. Confusion matrix (last epoch) ------------------------------
    try:
        if preds is not None and gts is not None:
            preds = np.array(preds)
            gts = np.array(gts)
            n_cls = max(preds.max(), gts.max()) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset} – Confusion Matrix (Last Epoch)")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()

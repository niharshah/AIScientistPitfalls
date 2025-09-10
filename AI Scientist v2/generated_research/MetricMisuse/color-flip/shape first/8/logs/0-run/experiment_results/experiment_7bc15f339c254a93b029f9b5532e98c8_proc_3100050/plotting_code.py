import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
#   set up paths and load data
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


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur else default


# ------------------------------------------------------------------ #
#   iterate over datasets
# ------------------------------------------------------------------ #
for dset, logs in experiment_data.items():
    # ------------- 1. loss curves ---------------------------------- #
    try:
        train_loss = logs.get("losses", {}).get("train", [])
        val_loss = logs.get("losses", {}).get("val", [])
        epochs = list(range(1, len(train_loss) + 1))
        if train_loss and val_loss:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset} – Train vs. Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset}: {e}")
        plt.close()

    # ------------- 2. CCWA curve ----------------------------------- #
    try:
        val_ccwa = logs.get("metrics", {}).get("val_CCWA", [])
        epochs = list(range(1, len(val_ccwa) + 1))
        if val_ccwa:
            plt.figure()
            plt.plot(epochs, val_ccwa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("CCWA")
            plt.title(f"{dset} – Validation CCWA Curve")
            fname = os.path.join(working_dir, f"{dset}_val_CCWA.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting CCWA for {dset}: {e}")
        plt.close()

    # ------------- 3. confusion matrix ----------------------------- #
    try:
        preds = logs.get("predictions", [])
        gts = logs.get("ground_truth", [])
        if preds and gts:
            y_pred = np.array(preds[-1])
            y_true = np.array(gts[-1])
            num_cls = max(y_true.max(), y_pred.max()) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset} – Confusion Matrix (last epoch)")
            for i in range(num_cls):
                for j in range(num_cls):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=7,
                    )
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()

    # ------------- print final metric ------------------------------ #
    final_ccwa = val_ccwa[-1] if val_ccwa else None
    print(f"{dset}: final Validation CCWA = {final_ccwa}")

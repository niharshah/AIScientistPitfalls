import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------------------------------------------------------
def safe_get(dct, *keys):
    cur = dct
    for k in keys:
        if cur is None or k not in cur:
            return None
        cur = cur[k]
    return cur


# iterate over datasets (usually just SPR_BENCH)
for dset_name, dset_dict in experiment_data.get("num_epochs", {}).items():
    # ------------------------------------------------------------------
    # 1) BWA curve
    try:
        train_bwa = safe_get(dset_dict, "metrics", "train")
        val_bwa = safe_get(dset_dict, "metrics", "val")
        if train_bwa and val_bwa:
            epochs = np.arange(1, len(train_bwa) + 1)
            plt.figure()
            plt.plot(epochs, train_bwa, label="Train BWA")
            plt.plot(epochs, val_bwa, label="Validation BWA")
            plt.xlabel("Epoch")
            plt.ylabel("Balanced Weighted Accuracy (BWA)")
            plt.title(f"{dset_name}: BWA over Epochs")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset_name}_bwa_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating BWA curve for {dset_name}: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 2) Loss curve
    try:
        train_loss = safe_get(dset_dict, "losses", "train")
        val_loss = safe_get(dset_dict, "losses", "val")
        if train_loss and val_loss:
            epochs = np.arange(1, len(train_loss) + 1)
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name}: Loss over Epochs")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset_name}: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 3) Confusion matrix on test predictions
    try:
        preds = dset_dict.get("predictions", [])
        gts = dset_dict.get("ground_truth", [])
        if preds and gts:
            preds = np.asarray(preds)
            gts = np.asarray(gts)
            num_classes = max(preds.max(), gts.max()) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"{dset_name}: Confusion Matrix (Test Set)")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

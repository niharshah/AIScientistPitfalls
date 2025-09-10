import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def safe_get(dct, *keys, default=None):
    for k in keys:
        dct = dct.get(k, {})
    return dct if dct else default


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset_name, d in experiment_data.items():
    epochs = range(1, len(d["losses"].get("train", [])) + 1)

    # 1) Loss curves
    try:
        tr_loss = d["losses"].get("train", [])
        val_loss = d["losses"].get("val", [])
        if tr_loss and val_loss:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name}: Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # 2) Weighted accuracy curves
    try:
        vals = d["metrics"].get("val", [])
        if vals:
            cwa = [m["cwa"] for m in vals]
            swa = [m["swa"] for m in vals]
            hmwa = [m["hmwa"] for m in vals]
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, hmwa, label="HMWA")
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(f"{dset_name}: Validation Weighted Accuracies")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_weighted_acc.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {dset_name}: {e}")
        plt.close()

    # 3) Confusion matrix (test)
    try:
        preds = np.array(d.get("predictions", []))
        gts = np.array(d.get("ground_truth", []))
        if preds.size and gts.size:
            classes = np.unique(np.concatenate([preds, gts]))
            cm = np.zeros((classes.size, classes.size), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset_name}: Confusion Matrix (Test Set)")
            plt.xticks(classes)
            plt.yticks(classes)
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()

            # Print evaluation metrics
            final_cwa = safe_get(d, "metrics", "val", -1, default={}).get("cwa", None)
            final_swa = safe_get(d, "metrics", "val", -1, default={}).get("swa", None)
            final_hmwa = safe_get(d, "metrics", "val", -1, default={}).get("hmwa", None)
            if final_hmwa is not None:
                print(
                    f"{dset_name} – Final Val HMWA: {final_hmwa:.4f}  "
                    f"CWA: {final_cwa:.4f}  SWA: {final_swa:.4f}"
                )
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

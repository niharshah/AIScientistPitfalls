import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------------
# iterate over datasets
for dset, logs in experiment_data.items():
    epochs = np.arange(1, len(logs.get("losses", {}).get("train", [])) + 1)

    # 1) Loss curve ----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, logs["losses"]["train"], label="Train Loss")
        plt.plot(epochs, logs["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset} Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved: {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # 2) Metric curve --------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, logs["metrics"]["train"], label="Train BWA")
        plt.plot(epochs, logs["metrics"]["val"], label="Val BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"{dset} Balanced Weighted Accuracy (BWA)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_bwa_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved: {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating BWA curve for {dset}: {e}")
        plt.close()

    # 3) Confusion matrix ---------------------------------------------
    try:
        preds = np.array(logs.get("predictions", []))
        gts = np.array(logs.get("ground_truth", []))
        if preds.size and gts.size:
            num_classes = max(preds.max(), gts.max()) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1

            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset} Confusion Matrix")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved: {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

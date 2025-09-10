import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

saved_files = []

for dset, info in experiment_data.items():
    # ---------- basic helpers -------------
    epochs = np.arange(1, len(info["losses"]["train"]) + 1)

    # ---------- 1. loss curves ------------
    try:
        plt.figure()
        plt.plot(epochs, info["losses"]["train"], label="train")
        plt.plot(epochs, info["losses"]["val"], "--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
        plt.savefig(fname, dpi=150)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ---------- 2. MCC curves -------------
    try:
        plt.figure()
        plt.plot(epochs, info["metrics"]["train_MCC"], label="train_MCC")
        plt.plot(epochs, info["metrics"]["val_MCC"], "--", label="val_MCC")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title(f"{dset}: Training vs Validation MCC")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_MCC_curves.png")
        plt.savefig(fname, dpi=150)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot for {dset}: {e}")
        plt.close()

    # ---------- 3. Confusion matrix -------
    try:
        preds = np.array(info["predictions"])
        gts = np.array(info["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title(f"{dset} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.colorbar()
        fname = os.path.join(working_dir, f"{dset}_conf_matrix.png")
        plt.savefig(fname, dpi=150)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

print("Saved figures:", saved_files)

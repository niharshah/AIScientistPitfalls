import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# pick dataset key
dkeys = list(experiment_data.keys())
if not dkeys:
    print("No experiment data found.")
    exit()
dset = dkeys[0]
data = experiment_data[dset]

epochs = list(range(1, len(data["metrics"]["train_loss"]) + 1))

# --------- 1. loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, data["metrics"]["train_loss"], label="Train")
    plt.plot(epochs, data["metrics"]["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dset} – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# --------- 2. AIS curve ----------
try:
    if data["metrics"]["AIS"]:
        plt.figure()
        plt.plot(epochs, data["metrics"]["AIS"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("AIS")
        plt.title(f"{dset} – Augmentation Invariance Score")
        fname = os.path.join(working_dir, f"{dset}_AIS_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating AIS plot: {e}")
    plt.close()

# --------- 3. confusion matrix ----------
try:
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(f"{dset} – Confusion Matrix (Dev)")
        fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# --------- print final accuracy ----------
if preds.size and gts.size:
    acc = (preds == gts).mean()
    print(f"Final dev accuracy: {acc:.3f}")

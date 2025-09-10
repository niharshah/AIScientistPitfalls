import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Only proceed if data is present
dname = "SPR_BENCH"
data = experiment_data.get(dname, None)

# --------- 1) Accuracy curves --------------------------------
try:
    if data:
        epochs = np.arange(1, len(data["metrics"]["train_acc"]) + 1)
        plt.figure()
        plt.plot(epochs, data["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, data["metrics"]["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy over Epochs\nDataset: {dname}")
        plt.legend()
        fpath = os.path.join(working_dir, f"{dname}_accuracy_curves.png")
        plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# --------- 2) Loss curves ------------------------------------
try:
    if data:
        epochs = np.arange(1, len(data["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Loss over Epochs\nDataset: {dname}")
        plt.legend()
        fpath = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------- 3) URA curve --------------------------------------
try:
    if data:
        epochs = np.arange(1, len(data["metrics"]["URA"]) + 1)
        plt.figure()
        plt.plot(epochs, data["metrics"]["URA"], marker="o", label="URA")
        plt.xlabel("Epoch")
        plt.ylabel("Unseen-Rule Accuracy")
        plt.title(f"URA over Epochs\nDataset: {dname}")
        plt.legend()
        fpath = os.path.join(working_dir, f"{dname}_URA_curve.png")
        plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating URA plot: {e}")
    plt.close()

# --------- 4) Confusion matrix (test set) --------------------
try:
    if data and data.get("ground_truth") and data.get("predictions"):
        gt = np.array(data["ground_truth"])
        pr = np.array(data["predictions"])
        labels = sorted(set(gt) | set(pr))
        lbl2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for g, p in zip(gt, pr):
            cm[lbl2idx[g], lbl2idx[p]] += 1

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_pos = np.arange(len(labels))
        plt.xticks(tick_pos, labels, rotation=90)
        plt.yticks(tick_pos, labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Confusion Matrix – Test Set\nDataset: {dname}")
        fpath = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

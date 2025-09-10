import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_key = "SPR_BENCH"
data = experiment_data.get(spr_key, {})


# ---------- helper for safe retrieval ----------
def safe_get(dic, *keys, default=None):
    for k in keys:
        dic = dic.get(k, {})
    return dic if dic else default


# ---------- Plot 1: Loss curve ----------
try:
    train_losses = safe_get(data, "losses", "train", default=[])
    val_losses = safe_get(data, "losses", "val", default=[])
    if train_losses and val_losses:
        plt.figure()
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{spr_key} Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{spr_key}_loss_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- Plot 2: Accuracy curve ----------
try:
    train_acc = safe_get(data, "metrics", "train", default=[])
    val_acc = safe_get(data, "metrics", "val", default=[])
    if train_acc and val_acc:
        plt.figure()
        epochs = np.arange(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{spr_key} Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{spr_key}_accuracy_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# ---------- Plot 3: Confusion Matrix ----------
try:
    preds = np.array(safe_get(data, "predictions", default=[]))
    gts = np.array(safe_get(data, "ground_truth", default=[]))
    if preds.size and gts.size and preds.shape == gts.shape:
        labels = np.unique(np.concatenate([preds, gts]))
        label2idx = {lab: i for i, lab in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for p, g in zip(preds, gts):
            cm[label2idx[g], label2idx[p]] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{spr_key} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.savefig(os.path.join(working_dir, f"{spr_key}_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Plotting complete. Files saved to:", working_dir)

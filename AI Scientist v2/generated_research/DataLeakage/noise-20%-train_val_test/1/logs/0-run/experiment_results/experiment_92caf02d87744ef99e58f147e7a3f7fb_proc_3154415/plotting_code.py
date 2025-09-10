import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate over datasets ----------
for dset_name, d in experiment_data.items():
    epochs = d.get("epochs", [])
    train_loss = d.get("losses", {}).get("train", [])
    val_loss = d.get("losses", {}).get("val", [])
    train_acc = d.get("metrics", {}).get("train_acc", [])
    val_acc = d.get("metrics", {}).get("val_acc", [])
    preds = np.array(d.get("predictions", []))
    gts = np.array(d.get("ground_truth", []))

    # ---- 1) loss curve ----
    try:
        if epochs and train_loss and val_loss:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name} Training vs Validation Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset_name}: {e}")
        plt.close()

    # ---- 2) accuracy curve ----
    try:
        if epochs and train_acc and val_acc:
            plt.figure()
            plt.plot(epochs, train_acc, label="Train Acc")
            plt.plot(epochs, val_acc, label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dset_name} Training vs Validation Accuracy")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset_name}_accuracy_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {dset_name}: {e}")
        plt.close()

    # ---- 3) confusion matrix ----
    try:
        if preds.size and gts.size:
            labels = np.unique(np.concatenate([gts, preds]))
            cm = np.zeros((labels.size, labels.size), dtype=int)
            for p, t in zip(preds, gts):
                cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(labels.size), labels, rotation=90)
            plt.yticks(range(labels.size), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset_name} Confusion Matrix (Test Set)")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

    # ---- evaluation metric ----
    if preds.size and gts.size:
        accuracy = (preds == gts).mean()
        print(f"{dset_name} Test Accuracy: {accuracy*100:.2f}%")

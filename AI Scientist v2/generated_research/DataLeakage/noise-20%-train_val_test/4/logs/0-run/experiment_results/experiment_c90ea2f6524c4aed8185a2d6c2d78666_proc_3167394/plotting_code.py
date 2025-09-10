import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# iterate over datasets (usually just 'SPR_MLM')
for dname in list(experiment_data.keys())[:5]:  # safety cap
    d = experiment_data[dname]
    epochs = np.asarray(d.get("epochs", []))
    if epochs.size == 0:
        continue  # nothing to plot

    # 1) Loss curves ----------------------------------------------------------
    try:
        plt.figure(figsize=(10, 4))
        # Left: training loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, d["losses"]["train"], label="train")
        plt.title(f"Left: Training Loss - {dname}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # Right: validation loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, d["losses"]["val"], label="val")
        plt.title(f"Right: Validation Loss - {dname}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curves for {dname}: {e}")
        plt.close()

    # 2) F1 curves ------------------------------------------------------------
    try:
        plt.figure(figsize=(10, 4))
        # Left: training F1
        plt.subplot(1, 2, 1)
        plt.plot(epochs, d["metrics"]["train_f1"], label="train")
        plt.title(f"Left: Training Macro-F1 - {dname}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        # Right: validation F1
        plt.subplot(1, 2, 2)
        plt.plot(epochs, d["metrics"]["val_f1"], label="val")
        plt.title(f"Right: Validation Macro-F1 - {dname}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dname}_f1_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating F1 curves for {dname}: {e}")
        plt.close()

    # 3) Confusion matrix on test set -----------------------------------------
    preds = np.asarray(d.get("predictions", []))
    gts = np.asarray(d.get("ground_truth", []))
    if preds.size and gts.size and preds.shape == gts.shape:
        try:
            labels = np.unique(np.concatenate([preds, gts]))
            cm = np.zeros((labels.size, labels.size), dtype=int)
            for p, t in zip(preds, gts):
                cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"Confusion Matrix - {dname}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(ticks=np.arange(labels.size), labels=labels, rotation=90)
            plt.yticks(ticks=np.arange(labels.size), labels=labels)
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating confusion matrix for {dname}: {e}")
            plt.close()

    # print final metric
    tf1 = d["metrics"].get("test_f1")
    if tf1 is not None:
        print(f"{dname} | Test Macro-F1: {tf1:.4f}")

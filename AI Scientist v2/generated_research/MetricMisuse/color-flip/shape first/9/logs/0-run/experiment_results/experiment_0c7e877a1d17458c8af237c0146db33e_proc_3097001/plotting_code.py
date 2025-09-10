import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper to pick dataset key
dname = next(iter(experiment_data.keys()), None)
if dname is None:
    print("No experiment data found.")
    exit()

data = experiment_data[dname]
epochs = range(1, len(data["losses"].get("train", [])) + 1)

# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    if data["losses"].get("train"):
        plt.plot(epochs, data["losses"]["train"], label="train")
    if data["losses"].get("val"):
        plt.plot(epochs, data["losses"]["val"], label="val")
    plt.title(f"{dname}: Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2: validation metric curve ----------
try:
    if data["metrics"].get("val"):
        plt.figure()
        plt.plot(
            epochs[: len(data["metrics"]["val"])], data["metrics"]["val"], marker="o"
        )
        plt.title(f"{dname}: Validation CWA-2D")
        plt.xlabel("Epoch")
        plt.ylabel("CWA-2D")
        fname = os.path.join(working_dir, f"{dname}_val_metric_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- plot 3: class distribution ----------
try:
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])
    if preds and gts:
        classes = sorted(set(gts + preds))
        pred_counts = [preds.count(c) for c in classes]
        gt_counts = [gts.count(c) for c in classes]

        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xticks(x, classes)
        plt.title(f"{dname}: Class Distribution (GT vs Pred)")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_class_distribution.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating distribution plot: {e}")
    plt.close()

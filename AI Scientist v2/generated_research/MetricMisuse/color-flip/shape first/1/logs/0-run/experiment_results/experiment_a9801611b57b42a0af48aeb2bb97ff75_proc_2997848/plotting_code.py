import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
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

ds_name = "SPR_BENCH"
ds = experiment_data.get(ds_name, {})

metrics = ds.get("metrics", {})
preds = np.array(ds.get("predictions", []))
gts = np.array(ds.get("ground_truth", []))

# ---------- 1. Train vs Val loss ----------
try:
    plt.figure()
    epochs = np.arange(1, len(metrics.get("train_loss", [])) + 1)
    plt.plot(epochs, metrics.get("train_loss", []), label="Train Loss")
    plt.plot(epochs, metrics.get("val_loss", []), label="Val Loss")
    plt.title(f"{ds_name}: Loss Curves\nTrain vs Validation")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error plotting loss curves: {e}")
    plt.close()

# ---------- 2. Weighted-accuracy curves ----------
try:
    plt.figure()
    for key, lab in zip(
        ["SWA", "CWA", "CompWA"],
        ["Shape-Wtd Acc", "Color-Wtd Acc", "Complex-Wtd Acc"],
    ):
        vals = metrics.get(key, [])
        if vals:
            plt.plot(range(1, len(vals) + 1), vals, label=lab)
    plt.title(f"{ds_name}: Weighted Accuracy Curves\nSWA / CWA / CompWA")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_weighted_accuracy_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error plotting accuracy curves: {e}")
    plt.close()

# ---------- 3. Confusion matrix ----------
try:
    if preds.size and gts.size:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title(f"{ds_name}: Confusion Matrix\nLeft: Ground Truth, Top: Predictions")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
        print("Confusion Matrix:\n", cm)
    else:
        print("No predictions/ground-truth found; skipping confusion matrix.")
except Exception as e:
    print(f"Error plotting confusion matrix: {e}")
    plt.close()

# ---------- print final metrics ----------
if metrics.get("SWA"):
    print(
        f"Final SWA={metrics['SWA'][-1]:.3f}, "
        f"CWA={metrics['CWA'][-1]:.3f}, "
        f"CompWA={metrics['CompWA'][-1]:.3f}"
    )

import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Assuming only one dataset present
if experiment_data:
    ds_name = list(experiment_data.keys())[0]
    ds = experiment_data[ds_name]

    # ------------------------------------------------------------------
    # Plot 1: Loss curves
    # ------------------------------------------------------------------
    try:
        plt.figure()
        epochs = np.arange(1, len(ds["losses"]["train"]) + 1)
        plt.plot(epochs, ds["losses"]["train"], label="Train")
        plt.plot(epochs, ds["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Accuracy curves
    # ------------------------------------------------------------------
    try:
        plt.figure()
        epochs = np.arange(1, len(ds["metrics"]["train"]) + 1)
        plt.plot(epochs, ds["metrics"]["train"], label="Train")
        plt.plot(epochs, ds["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} – Training vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Confusion-matrix heat map (optional)
    # ------------------------------------------------------------------
    try:
        preds = np.array(ds.get("predictions", []))
        gts = np.array(ds.get("ground_truth", []))
        if preds.size and gts.size:
            num_classes = int(max(gts.max(), preds.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.title(f"{ds_name} – Confusion Matrix (Test Split)")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Print evaluation metrics
    # ------------------------------------------------------------------
    test_acc = ds["metrics"].get("test_acc", None)
    ura = ds["metrics"].get("URA", None)
    if test_acc is not None:
        print(f"Test Accuracy ({ds_name}): {test_acc:.3f}")
    if ura is not None:
        print(f"Unseen-Rule Accuracy ({ds_name}): {ura:.3f}")
else:
    print("No experiment data found; skipping plots.")

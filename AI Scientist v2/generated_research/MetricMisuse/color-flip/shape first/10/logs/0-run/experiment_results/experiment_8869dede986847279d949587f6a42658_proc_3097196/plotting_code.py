import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_BENCH"
ed = experiment_data.get(dataset_name, {})

# Helper: get epochs axis
train_losses = ed.get("losses", {}).get("train", [])
val_losses = ed.get("losses", {}).get("val", [])
epochs = np.arange(1, len(train_losses) + 1)

# ------------------------------- Plot 1 ------------------------------------- #
try:
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset_name} Loss Curve")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name.lower()}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------- Plot 2 ------------------------------------- #
try:
    swa = ed.get("metrics", {}).get("val_swa", [])
    cwa = ed.get("metrics", {}).get("val_cwa", [])
    hwa = ed.get("metrics", {}).get("val_hwa", [])
    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, hwa, label="HWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title(f"{dataset_name} Validation Weighted Accuracies")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name.lower()}_val_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics curve: {e}")
    plt.close()

# ------------------------------- Plot 3 ------------------------------------- #
try:
    preds = np.array(ed.get("predictions", []), dtype=int)
    gts = np.array(ed.get("ground_truth", []), dtype=int)
    labels = np.unique(np.concatenate([preds, gts]))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1  # rows: true, cols: pred
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks(labels, labels)
    plt.yticks(labels, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"{dataset_name} Confusion Matrix\nLeft: True Label, Right: Prediction")
    fname = os.path.join(working_dir, f"{dataset_name.lower()}_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ----------------------------- Print metrics -------------------------------- #
try:
    test_swa = ed.get("metrics", {}).get("test_swa", None)
    test_cwa = ed.get("metrics", {}).get("test_cwa", None)
    test_hwa = ed.get("metrics", {}).get("test_hwa", None)
    print(f"Test SWA: {test_swa:.4f}, CWA: {test_cwa:.4f}, HWA: {test_hwa:.4f}")
except Exception as e:
    print(f"Error printing metrics: {e}")

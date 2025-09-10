import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

from sklearn.metrics import confusion_matrix, f1_score

ds_name = "SPR_BENCH"
data = experiment_data.get(ds_name, {})

epochs = data.get("epochs", [])
train_l = data.get("losses", {}).get("train", [])
val_l = data.get("losses", {}).get("val", [])
train_f1 = data.get("metrics", {}).get("train_f1", [])
val_f1 = data.get("metrics", {}).get("val_f1", [])
preds = data.get("predictions", [])
gts = data.get("ground_truth", [])

# Plot 1: Loss curves
try:
    if epochs and train_l and val_l:
        plt.figure()
        plt.plot(epochs, train_l, label="Train Loss")
        plt.plot(epochs, val_l, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# Plot 2: F1 curves
try:
    if epochs and train_f1 and val_f1:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{ds_name} Macro-F1 Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_f1_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# Plot 3: Confusion matrix
try:
    if preds and gts:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{ds_name} Confusion Matrix\nDataset: Test Split")
        fname = os.path.join(working_dir, f"{ds_name.lower()}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# Print overall Macro-F1 on test set if available
if preds and gts:
    print("Test Macro-F1:", f1_score(gts, preds, average="macro"))

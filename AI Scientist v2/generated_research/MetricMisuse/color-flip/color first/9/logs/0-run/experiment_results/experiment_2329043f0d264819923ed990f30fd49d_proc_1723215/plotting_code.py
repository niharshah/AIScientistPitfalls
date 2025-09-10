import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
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

# early exit if nothing to plot
if not experiment_data:
    quit()

# we assume a single dataset entry
dataset_name = next(iter(experiment_data))
data = experiment_data[dataset_name]


# ---------- helper to compute accuracy ----------
def simple_accuracy(gt, pr):
    gt = np.array(gt)
    pr = np.array(pr)
    return float(np.mean(gt == pr)) if gt.size else 0.0


# ---------- PLOT 1: losses ----------
try:
    epochs, train_loss = zip(*data["losses"]["train"])
    _, val_loss = (
        zip(*data["losses"].get("val", [])) if data["losses"].get("val") else ([], [])
    )
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    subtitle = "Train vs Val Loss" if val_loss else "Train Loss"
    plt.title(f"{dataset_name} Loss Curve\n{subtitle}")
    plt.legend()
    fname = f"{dataset_name.lower()}_loss_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- PLOT 2: validation metrics ----------
try:
    if data["metrics"]["val"]:
        epochs, cwa, swa, dwhs = zip(*data["metrics"]["val"])
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, dwhs, label="DWHS")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dataset_name} Validation Metrics\nCWA, SWA, DWHS over Epochs")
        plt.legend()
        fname = f"{dataset_name.lower()}_val_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------- final test accuracy ----------
try:
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    acc = simple_accuracy(gts, preds)
    print(f"Test accuracy based on saved predictions: {acc:.4f}")
except Exception as e:
    print(f"Error computing test accuracy: {e}")

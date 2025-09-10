import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
data = experiment_data.get(ds_key, {})


# -------- helper --------
def get_epochs_and_vals(tuples):
    if not tuples:
        return [], []
    epochs, vals = zip(*tuples)
    return list(epochs), list(vals)


# -------- 1) Loss curves --------
try:
    plt.figure()
    train_epochs, train_losses = get_epochs_and_vals(
        data.get("losses", {}).get("train", [])
    )
    val_epochs, val_losses = get_epochs_and_vals(data.get("losses", {}).get("val", []))
    if train_epochs:
        plt.plot(train_epochs, train_losses, label="Train")
    if val_epochs:
        plt.plot(val_epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_key} Loss Curves\nLeft: Training, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_key}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------- 2) Validation metrics --------
try:
    val_metrics_entries = data.get("metrics", {}).get("val", [])
    if val_metrics_entries:
        epochs = [e for e, _ in val_metrics_entries]
        acc = [d["acc"] for _, d in val_metrics_entries]
        pcwa = [d["pcwa"] for _, d in val_metrics_entries]
        cwa = [d["cwa"] for _, d in val_metrics_entries]
        swa = [d["swa"] for _, d in val_metrics_entries]

        plt.figure()
        plt.plot(epochs, acc, label="Acc")
        plt.plot(epochs, pcwa, label="PCWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title(f"{ds_key} Validation Metrics Across Epochs\nAcc / PCWA / CWA / SWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_val_metrics.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# -------- 3) Confusion matrix (final) --------
try:
    y_pred = np.array(data.get("predictions", []))
    y_true = np.array(data.get("ground_truth", []))
    if y_pred.size and y_true.size:
        num_labels = int(max(y_true.max(), y_pred.max())) + 1
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"{ds_key} Confusion Matrix\nCounts per Class")
        fname = os.path.join(working_dir, f"{ds_key}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------- print latest metrics --------
if data.get("metrics", {}).get("val"):
    last_epoch, last_metrics = data["metrics"]["val"][-1]
    print(f"Final Validation Metrics at epoch {last_epoch}: {last_metrics}")

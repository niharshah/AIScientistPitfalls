import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
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

# assume only one dataset entry (per provided code)
dataset_key = next(iter(experiment_data.keys()), None)
if dataset_key is None:
    print("No experiment data found.")
    exit()

dset = experiment_data[dataset_key]["dataset"]
epochs = dset["epochs"]
loss_train, loss_val = dset["losses"]["train"], dset["losses"]["val"]
metrics = dset["metrics"]
preds, gts = np.array(dset.get("predictions", [])), np.array(
    dset.get("ground_truth", [])
)

# 1. Loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_key}: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_key}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()


# helper to plot each metric curve
def plot_metric(metric_name):
    try:
        plt.figure()
        plt.plot(epochs, metrics["train"][metric_name], label="Train")
        plt.plot(epochs, metrics["val"][metric_name], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{dataset_key}: {metric_name} over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_{metric_name}_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {metric_name} curve: {e}")
        plt.close()


# 2-4. Metric curves
for m in ["CWA", "SWA", "CmpWA"]:
    plot_metric(m)

# 5. Confusion matrix (if predictions exist)
try:
    if preds.size and gts.size:
        classes = sorted(set(gts) | set(preds))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dataset_key}: Confusion Matrix")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="red" if cm[i, j] else "black",
                )
        fname = os.path.join(working_dir, f"{dataset_key}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Predictions / ground truth not found – skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# print stored test metrics
test_metrics = experiment_data[dataset_key]["dataset"].get("test_metrics", {})
print("Test metrics:", test_metrics)

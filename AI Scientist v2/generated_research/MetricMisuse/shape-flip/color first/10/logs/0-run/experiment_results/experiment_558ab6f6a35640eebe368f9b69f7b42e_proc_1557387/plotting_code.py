import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch nested dicts safely
def get(*keys, default=None):
    d = experiment_data
    for k in keys:
        d = d.get(k, {})
    return d if d else default


abl = "NoSeqEdge"
dset = "SPR_BENCH"
loss_train = get(abl, dset, "losses", "train", default=[])
loss_val = get(abl, dset, "losses", "val", default=[])
val_metrics = get(abl, dset, "metrics", "val", default=[])
test_metrics = get(abl, dset, "metrics", "test", default={})

# ------------------------------------------------------------------
# 1) Train / Val loss curve
try:
    if loss_train and loss_val:
        epochs = np.arange(1, len(loss_train) + 1)
        plt.figure()
        plt.plot(epochs, loss_train, label="Train")
        plt.plot(epochs, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{abl} - {dset}\nTraining vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{abl}_{dset}_loss_curve.png")
        plt.savefig(fname)
    else:
        print("Loss data not found, skipping loss curve.")
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# 2) Validation metric curves
try:
    if val_metrics:
        epochs = [m["epoch"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        swa = [m["swa"] for m in val_metrics]
        cpxw = [m["cpxwa"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cpxw, label="CpxWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(f"{abl} - {dset}\nValidation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, f"{abl}_{dset}_val_metrics.png")
        plt.savefig(fname)
    else:
        print("Validation metrics not found, skipping val curves.")
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# 3) Test metric bar chart
try:
    if test_metrics:
        labels = ["CWA", "SWA", "CpxWA"]
        values = [
            test_metrics.get("cwa", 0),
            test_metrics.get("swa", 0),
            test_metrics.get("cpxwa", 0),
        ]
        plt.figure()
        plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title(f"{abl} - {dset}\nTest Weighted Accuracies")
        fname = os.path.join(working_dir, f"{abl}_{dset}_test_metrics.png")
        plt.savefig(fname)
    else:
        print("Test metrics not found, skipping test bar chart.")
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
finally:
    plt.close()

# ------------------------------------------------------------------
# print numeric test results
if test_metrics:
    print("Test metrics:", test_metrics)

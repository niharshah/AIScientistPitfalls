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


# helper: extract (epoch,value) tuples -> np arrays
def to_xy(tuples):
    if not tuples:
        return np.array([]), np.array([])
    x, y = zip(*tuples)
    return np.asarray(x), np.asarray(y)


ds_name = "default_spr"
ds = experiment_data.get(ds_name, {})

# ------------- Plot 1: loss curves -------------
try:
    tr_epochs, tr_loss = to_xy(ds.get("losses", {}).get("train", []))
    va_epochs, va_loss = to_xy(ds.get("losses", {}).get("val", []))
    if tr_epochs.size and va_epochs.size:
        plt.figure()
        plt.plot(tr_epochs, tr_loss, label="Train Loss")
        plt.plot(va_epochs, va_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------- Plot 2: val accuracy, CWA, SWA -------------
try:
    metrics = ds.get("metrics", {}).get("val", [])
    if metrics:
        epochs = np.array([m["epoch"] for m in metrics])
        acc = np.array([m["acc"] for m in metrics])
        cwa = np.array([m["cwa"] for m in metrics])
        swa = np.array([m["swa"] for m in metrics])

        plt.figure()
        plt.plot(epochs, acc, label="Accuracy")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Validation Accuracy Metrics")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_val_accuracy_metrics.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy metric plot: {e}")
    plt.close()

# ------------- Plot 3: JWA -------------
try:
    metrics = ds.get("metrics", {}).get("val", [])
    if metrics:
        epochs = np.array([m["epoch"] for m in metrics])
        jwa = np.array([m["jwa"] for m in metrics])

        plt.figure()
        plt.plot(epochs, jwa, label="JWA", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("JWA")
        plt.title(f"{ds_name}: Joint Weighted Accuracy (JWA)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_val_jwa.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating JWA plot: {e}")
    plt.close()

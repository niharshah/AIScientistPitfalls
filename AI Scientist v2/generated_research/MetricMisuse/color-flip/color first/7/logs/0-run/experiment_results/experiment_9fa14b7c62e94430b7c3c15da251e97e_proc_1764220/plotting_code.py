import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------------------------------------------------------
def get_ed():
    try:
        return experiment_data["BagOfEmbeddings"]["SPR_BENCH"]
    except Exception as e:
        print(f"Unexpected structure in experiment_data: {e}")
        return None


ed = get_ed()
if ed is None:
    exit()

epochs = ed.get("epochs", [])
train_losses = ed.get("losses", {}).get("train", [])
train_metrics = ed.get("metrics", {}).get("train", [])
val_metrics = ed.get("metrics", {}).get("val", [])


# Helper to pull metric series safely
def metric_series(name, split_metrics):
    return [m.get(name, np.nan) for m in split_metrics]


# ------------------------------------------------------------------
# 1) Train vs Val Loss
try:
    if train_losses:
        plt.figure()
        plt.plot(epochs, train_losses, marker="o", label="Train Loss")
        plt.title("SPR_BENCH: Training Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Complexity-Weighted Accuracy
try:
    if val_metrics:
        plt.figure()
        plt.plot(epochs, metric_series("cpx", train_metrics), label="Train CpxWA")
        plt.plot(epochs, metric_series("cpx", val_metrics), label="Val CpxWA")
        plt.title("SPR_BENCH: Complexity-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_cpxwa_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CpxWA curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Color & Shape Weighted Accuracy
try:
    if val_metrics:
        plt.figure()
        plt.plot(epochs, metric_series("cwa", train_metrics), label="Train CWA")
        plt.plot(epochs, metric_series("cwa", val_metrics), label="Val CWA")
        plt.plot(epochs, metric_series("swa", train_metrics), label="Train SWA")
        plt.plot(epochs, metric_series("swa", val_metrics), label="Val SWA")
        plt.title("SPR_BENCH: Color & Shape Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_cwa_swa_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CWA/SWA curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4) Confusion Matrix at Best Epoch
try:
    from matplotlib import cm

    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])
    if preds and gts and len(preds) == len(gts):
        num_classes = len(set(gts) | set(preds))
        cmatrix = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, gts):
            cmatrix[t, p] += 1
        plt.figure()
        plt.imshow(cmatrix, interpolation="nearest", cmap=cm.Blues)
        plt.colorbar()
        plt.title("SPR_BENCH: Confusion Matrix (Best Val Epoch)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

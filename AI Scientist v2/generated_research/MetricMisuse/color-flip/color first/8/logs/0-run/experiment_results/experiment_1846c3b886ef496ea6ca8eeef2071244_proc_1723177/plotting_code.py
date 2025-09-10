import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
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


# helper to fetch safely
def get(ds, *keys, default=None):
    cur = experiment_data.get(ds, {})
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur else default


ds_name = "spr_bench"

# ------------------ Plot 1: Loss curves ------------------
try:
    losses_train = get(ds_name, "losses", "train", default=[])
    losses_val = get(ds_name, "losses", "val", default=[])
    if losses_train and losses_val:
        epochs_t, train_vals = zip(*losses_train)
        epochs_v, val_vals = zip(*losses_val)
        plt.figure()
        plt.plot(epochs_t, train_vals, label="Train")
        plt.plot(epochs_v, val_vals, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench: Training vs. Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------ Plot 2: Validation metrics ------------
try:
    metrics_val = get(ds_name, "metrics", "val", default=[])
    if metrics_val:
        ep, cwa, swa, cshm = zip(*metrics_val)
        plt.figure()
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, cshm, label="CSHM")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("spr_bench: Validation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_weighted_accuracies.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# -------------- Plot 3: Confusion Matrix -----------------
try:
    preds = np.array(get(ds_name, "predictions", default=[]))
    gts = np.array(get(ds_name, "ground_truth", default=[]))
    if preds.size and gts.size:
        n_classes = max(preds.max(), gts.max()) + 1
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("spr_bench: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ----------------- Evaluation Metric ---------------------
if preds.size and gts.size:
    test_acc = (preds == gts).mean()
    print(f"Test Accuracy: {test_acc:.4f}")

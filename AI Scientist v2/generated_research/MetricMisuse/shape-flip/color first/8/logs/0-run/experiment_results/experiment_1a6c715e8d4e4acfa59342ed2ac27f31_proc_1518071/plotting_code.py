import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested dicts
def get_nested(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


abl = "token_level"
dset = "SPR_BENCH"
loss_tr = get_nested(experiment_data, abl, dset, "losses", "train", default=[])
loss_val = get_nested(experiment_data, abl, dset, "losses", "val", default=[])
acc_tr = get_nested(experiment_data, abl, dset, "metrics", "train", default=[])
acc_val = get_nested(experiment_data, abl, dset, "metrics", "val", default=[])
preds = get_nested(experiment_data, abl, dset, "predictions", default=[])
gts = get_nested(experiment_data, abl, dset, "ground_truth", default=[])

# ------------- PLOTS -------------
# 1) Loss curve
try:
    plt.figure()
    epochs = np.arange(1, len(loss_tr) + 1)
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dset} – Loss Curve (Train vs Validation)")
    plt.legend()
    fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Accuracy curve
try:
    plt.figure()
    epochs = np.arange(1, len(acc_tr) + 1)
    plt.plot(epochs, acc_tr, label="Train")
    plt.plot(epochs, acc_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dset} – Accuracy Curve (Train vs Validation)")
    plt.legend()
    fname = os.path.join(working_dir, f"{dset}_accuracy_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 3) Confusion matrix
try:
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        labels = sorted(set(gts))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xticks(labels)
        plt.yticks(labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{dset} – Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("No predictions / ground truth found, skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------- METRIC PRINTS -------------
if acc_val:
    print(f"Final validation accuracy: {acc_val[-1]:.3f}")
if preds and gts:
    print("Confusion matrix counts:\n", cm)

import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

# ---------------------------------------------------------------------
# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

bench = experiment_data.get("SPR")
if bench is None:
    print("SPR dataset not found in experiment_data.npy")
    exit()

# ---------------------------------------------------------------------
# pull arrays
train_loss = np.asarray(bench["losses"]["train"])
val_loss = np.asarray(bench["losses"]["val"])
epochs = np.arange(1, len(train_loss) + 1)

swa = np.asarray([m["swa"] for m in bench["metrics"]["val"]])
cwa = np.asarray([m["cwa"] for m in bench["metrics"]["val"]])
acs = np.asarray([m["acs"] for m in bench["metrics"]["val"]])

preds = np.asarray(bench["predictions"])
gts = np.asarray(bench["ground_truth"])
test_acs = float(bench["acs"][0]) if bench["acs"] else np.nan

# ---------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Synthetic Dataset: Training vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) Metric curves
try:
    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, acs, label="ACS")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR Synthetic Dataset: Weighted Accuracy Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "spr_metric_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Confusion matrix
try:
    cm = np.zeros((2, 2), dtype=int)
    for gt, pr in zip(gts, preds):
        cm[gt, pr] += 1

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i, j in product(range(2), range(2)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("SPR Synthetic Dataset: Confusion Matrix (Test)")
    plt.colorbar()
    fname = os.path.join(working_dir, "spr_confusion_matrix.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# print final metrics
test_acc = (gts == preds).mean() if len(gts) else np.nan
print(f"Test Accuracy: {test_acc:.3f} | Test ACS: {test_acs:.3f}")

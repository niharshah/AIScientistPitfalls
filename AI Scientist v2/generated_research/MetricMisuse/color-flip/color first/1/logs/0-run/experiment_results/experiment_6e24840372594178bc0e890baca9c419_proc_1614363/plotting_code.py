import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

# ---------------- paths / load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp, tag = None, None

tag = "no_transformer_mean_pool"
ds = "SPR"
if exp is None or tag not in exp or ds not in exp[tag]:
    print("Required experiment information missing, nothing to plot.")
    exit()

edata = exp[tag][ds]
loss_tr = edata["losses"].get("train", [])
loss_val = edata["losses"].get("val", [])
metrics_val = edata["metrics"].get("val", [])
metrics_test = edata["metrics"].get("test", {})
preds = np.array(edata.get("predictions", []))
gts = np.array(edata.get("ground_truth", []))


# -------- helper to get series from list of dicts
def ser(key):
    return [m.get(key, np.nan) for m in metrics_val]


# -------------------- PLOT 1 : loss curves
try:
    epochs = range(1, max(len(loss_tr), len(loss_val)) + 1)
    plt.figure()
    if loss_tr:
        plt.plot(epochs, loss_tr, label="Train")
    if loss_val:
        plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Dataset – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------- PLOT 2 : CVA curve
try:
    cva = ser("cva")
    if any(~np.isnan(cva)):
        plt.figure()
        plt.plot(range(1, len(cva) + 1), cva, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Composite Variety Accuracy")
        plt.title("SPR Dataset – Validation CVA over Epochs")
        fname = os.path.join(working_dir, "SPR_CVA_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CVA plot: {e}")
    plt.close()

# -------------------- PLOT 3 : CWA & SWA curves
try:
    cwa, swa = ser("cwa"), ser("swa")
    if any(~np.isnan(cwa)) or any(~np.isnan(swa)):
        plt.figure()
        if any(~np.isnan(cwa)):
            plt.plot(range(1, len(cwa) + 1), cwa, label="CWA")
        if any(~np.isnan(swa)):
            plt.plot(range(1, len(swa) + 1), swa, label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR Dataset – Validation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_weighted_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()

# -------------------- PLOT 4 : Test metric bar chart
try:
    if metrics_test:
        keys, vals = zip(*[(k.upper(), v) for k, v in metrics_test.items()])
        plt.figure()
        plt.bar(keys, vals)
        plt.ylim(0, 1)
        plt.title("SPR Dataset – Test Set Metrics")
        fname = os.path.join(working_dir, "SPR_test_metrics_bar.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating test metrics bar plot: {e}")
    plt.close()

# -------------------- PLOT 5 : Confusion matrix
try:
    if preds.size and gts.size:
        n_cls = int(max(max(preds), max(gts)) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR Dataset – Confusion Matrix (Test)")
        # annotate cells for readability
        for i, j in product(range(n_cls), range(n_cls)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------------- print test metrics
if metrics_test:
    print("Test Metrics:", metrics_test)

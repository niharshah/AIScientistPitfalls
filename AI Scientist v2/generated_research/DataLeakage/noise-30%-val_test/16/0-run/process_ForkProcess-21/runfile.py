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

# detect ablation and dataset names automatically
if experiment_data:
    abl_name = list(experiment_data.keys())[0]
    ds_name = list(experiment_data[abl_name].keys())[0]
    ed = experiment_data[abl_name][ds_name]
else:
    abl_name = ds_name = ""
    ed = {}


# helper: extract arrays if they exist
def _get_metric(split, key):
    return [m.get(key, np.nan) for m in ed["metrics"].get(split, [])]


epochs = ed.get("epochs", [])
train_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
train_acc = _get_metric("train", "acc")
val_acc = _get_metric("val", "acc")
train_mcc = _get_metric("train", "MCC")
val_mcc = _get_metric("val", "MCC")
train_rma = _get_metric("train", "RMA")
val_rma = _get_metric("val", "RMA")

preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# ------------------------------ PLOTS ------------------------------
# 1. Loss curve
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ds_name} Loss Curve")
    plt.legend()
    fname = f"{ds_name}_loss_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. Accuracy curve
try:
    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{ds_name} Accuracy Curve")
    plt.legend()
    fname = f"{ds_name}_accuracy_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 3. MCC curve
try:
    plt.figure()
    plt.plot(epochs, train_mcc, label="Train")
    plt.plot(epochs, val_mcc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title(f"{ds_name} MCC Curve")
    plt.legend()
    fname = f"{ds_name}_mcc_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve: {e}")
    plt.close()

# 4. Rule-Macro Accuracy curve
try:
    plt.figure()
    plt.plot(epochs, train_rma, label="Train")
    plt.plot(epochs, val_rma, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Rule-Macro Accuracy")
    plt.title(f"{ds_name} RMA Curve")
    plt.legend()
    fname = f"{ds_name}_rma_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating RMA curve: {e}")
    plt.close()

# 5. Confusion matrix on test set
try:
    if preds.size and gts.size:
        tp = int(((preds == 1) & (gts == 1)).sum())
        fp = int(((preds == 1) & (gts == 0)).sum())
        fn = int(((preds == 0) & (gts == 1)).sum())
        tn = int(((preds == 0) & (gts == 0)).sum())
        cm = np.array([[tp, fp], [fn, tn]])
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 1", "Pred 0"])
        plt.yticks([0, 1], ["True 1", "True 0"])
        plt.title(f"{ds_name} Confusion Matrix")
        fname = f"{ds_name}_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------ METRIC PRINT -----------------------
# Print stored test metrics if present
test_metrics = ed.get("test_metrics", {})
if test_metrics:
    print("=== Stored Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

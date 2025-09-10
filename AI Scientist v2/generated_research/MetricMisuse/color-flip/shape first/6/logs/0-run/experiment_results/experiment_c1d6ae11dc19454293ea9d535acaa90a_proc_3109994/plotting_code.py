import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
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
# helper to safely fetch the SPR sub-dict
def get_spr(exp_dict):
    for ablation in exp_dict.values():  # e.g. 'TokenDropoutOnly'
        if "SPR" in ablation:
            return ablation["SPR"]
    return None


spr = get_spr(experiment_data)
if spr is None:
    print("SPR dataset entry not found in experiment_data.")
    exit()

epochs = spr["epochs"]
loss_tr = spr["losses"]["train"]
loss_val = spr["losses"]["val"]
metrics_val = spr["metrics"]["val"]  # list of dicts
preds = spr["predictions"]
gts = spr["ground_truth"]

# ------------------------------------------------------------------
# 1. Loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR: Training vs. Validation Loss")
    plt.legend()
    fpath = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2. Validation metrics curves
try:
    swa = [m["SWA"] for m in metrics_val]
    cwa = [m["CWA"] for m in metrics_val]
    scaa = [m["SCAA"] for m in metrics_val]

    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, scaa, label="SCAA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR: Validation Metrics Over Epochs")
    plt.legend()
    fpath = os.path.join(working_dir, "SPR_validation_metrics.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3. Confusion matrix (final epoch)
try:
    classes = sorted(set(gts) | set(preds))
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(n_cls), classes)
    plt.yticks(range(n_cls), classes)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(
        "SPR: Confusion Matrix (Final Epoch)\nLeft: Ground Truth, Right: Predictions"
    )
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fpath = os.path.join(working_dir, "SPR_confusion_matrix.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Final numeric summary
if metrics_val:
    final_metrics = metrics_val[-1]
    print(
        f"Final Validation Metrics - SCAA: {final_metrics['SCAA']:.3f}, "
        f"SWA: {final_metrics['SWA']:.3f}, CWA: {final_metrics['CWA']:.3f}"
    )

import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------------
# setup & data loading
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_key = "SPR_BENCH"
ed = experiment_data.get(data_key, {}) if experiment_data else {}


# -------------------------------------------------------------------------
# helper: fetch series safely
def _series(split, key):
    lst = ed.get(key, {}).get(split, [])
    epochs = [d["epoch"] for d in lst]
    vals = [d[next(k for k in d if k != "epoch")] for d in lst]
    return epochs, vals


# 1) Loss curve ------------------------------------------------------------
try:
    tr_ep, tr_loss = _series("train", "losses")
    val_ep, val_loss = _series("val", "losses")
    if tr_ep and val_ep:
        plt.figure()
        plt.plot(tr_ep, tr_loss, label="Train")
        plt.plot(val_ep, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{data_key} Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{data_key}_loss_curve.png")
        plt.savefig(fname)
    else:
        print("Loss data missing, skipping loss plot.")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Macro-F1 curve --------------------------------------------------------
try:
    tr_ep, tr_f1 = _series("train", "metrics")
    val_ep, val_f1 = _series("val", "metrics")
    if tr_ep and val_ep:
        plt.figure()
        plt.plot(tr_ep, tr_f1, label="Train")
        plt.plot(val_ep, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(f"{data_key} Macro-F1 Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{data_key}_f1_curve.png")
        plt.savefig(fname)
    else:
        print("Metric data missing, skipping F1 plot.")
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# 3) Confusion matrix ------------------------------------------------------
try:
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            f"{data_key} Confusion Matrix\nLeft: Ground Truth, Right: Predicted Samples"
        )
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="red", fontsize=8
                )
        fname = os.path.join(working_dir, f"{data_key}_confusion_matrix.png")
        plt.savefig(fname)
    else:
        print("Prediction data missing, skipping confusion matrix.")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------------------------------------------------------------
# Print evaluation metric
if ed.get("predictions") and ed.get("ground_truth"):
    from sklearn.metrics import f1_score

    test_f1 = f1_score(ed["ground_truth"], ed["predictions"], average="macro")
    print(f"Test Macro_F1: {test_f1:.3f}")

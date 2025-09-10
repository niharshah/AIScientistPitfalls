import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
if ds_key not in experiment_data:
    print(f"Dataset {ds_key} not found in experiment data.")
    exit()

ds = experiment_data[ds_key]

# -------------------- 1. Loss curves --------------------
try:
    epochs = ds["epochs"]
    tr_loss = ds["losses"]["train"]
    val_loss = ds["losses"]["val"]
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------- 2. Macro-F1 curves --------------------
try:
    tr_f1 = ds["metrics"]["train_macro_f1"]
    val_f1 = ds["metrics"]["val_macro_f1"]
    plt.figure()
    plt.plot(epochs, tr_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 plot: {e}")
    plt.close()

# -------------------- 3. Confusion matrix --------------------
try:
    preds = np.array(ds.get("predictions", []))
    gts = np.array(ds.get("ground_truth", []))
    if preds.size > 0 and gts.size > 0:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix\nDataset: Test Set")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Predictions / ground-truth empty; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------------------- print evaluation metric --------------------
if preds.size > 0 and gts.size > 0:
    test_macro_f1 = f1_score(gts, preds, average="macro")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")

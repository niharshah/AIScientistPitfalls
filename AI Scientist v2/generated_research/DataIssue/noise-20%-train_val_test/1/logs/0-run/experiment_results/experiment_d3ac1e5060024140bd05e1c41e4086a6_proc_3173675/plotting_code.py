import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

saved_files = []
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to get nested dict safely
def _get(dic, *keys, default=None):
    for k in keys:
        if dic is None:
            return default
        dic = dic.get(k, None)
    return dic if dic is not None else default


ed = _get(experiment_data, "no_positional_encoding", "SPR_BENCH", default={})
metrics = ed.get("metrics", {})
losses = ed.get("losses", {})

# Plot 1: Loss curves ---------------------------------------------------------
try:
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if train_loss and val_loss:
        plt.figure()
        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Validation")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        saved_files.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 2: Accuracy curves ------------------------------------------------------
try:
    tr_acc = metrics.get("train_acc", [])
    va_acc = metrics.get("val_acc", [])
    if tr_acc and va_acc:
        plt.figure()
        plt.plot(tr_acc, label="Train")
        plt.plot(va_acc, label="Validation")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        saved_files.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 3: Macro-F1 curves ------------------------------------------------------
try:
    tr_f1 = metrics.get("train_f1", [])
    va_f1 = metrics.get("val_f1", [])
    if tr_f1 and va_f1:
        plt.figure()
        plt.plot(tr_f1, label="Train")
        plt.plot(va_f1, label="Validation")
        plt.title("SPR_BENCH Macro-F1 Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        saved_files.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# Plot 4: Confusion matrix -----------------------------------------------------
try:
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])
    if len(preds) and len(gts):
        preds = np.asarray(preds, dtype=int)
        gts = np.asarray(gts, dtype=int)
        num_cls = int(preds.max() + 1)
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        saved_files.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Saved figures:", saved_files)

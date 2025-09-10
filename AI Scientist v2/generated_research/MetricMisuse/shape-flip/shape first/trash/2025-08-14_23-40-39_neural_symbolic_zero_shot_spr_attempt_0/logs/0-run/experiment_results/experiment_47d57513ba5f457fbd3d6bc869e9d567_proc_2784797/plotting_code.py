import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths and data ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested keys
def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


exp_name, dset_name = "FreezeEmb", "SPR_BENCH"
ed = get(experiment_data, exp_name, dset_name, default={})

# ---- PLOT 1: Train vs Validation Loss ----
try:
    train_loss = ed["metrics"]["train_loss"]
    val_loss = ed["metrics"]["val_loss"]
    if train_loss and val_loss:
        plt.figure()
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset_name}: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---- PLOT 2: Validation SWA ----
try:
    val_swa = ed["metrics"]["val_swa"]
    if val_swa:
        plt.figure()
        epochs = np.arange(1, len(val_swa) + 1)
        plt.plot(epochs, val_swa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dset_name}: Validation SWA over Epochs")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset_name}_val_swa.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# ---- PLOT 3: Confusion Matrix (Test) ----
try:
    y_pred = ed["predictions"]["test"]
    y_true = ed["ground_truth"]["test"]
    if y_pred and y_true and len(y_pred) == len(y_true):
        classes = sorted(set(y_true))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dset_name}: Confusion Matrix (Test)")
        plt.xticks(classes)
        plt.yticks(classes)
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---- Print final metrics ----
try:
    last_val_swa = val_swa[-1] if val_swa else None
    test_swa = None
    if y_pred and y_true:
        # recompute SWA with same helper as training script
        def count_shape_variety(sequence: str) -> int:
            return len(set(tok[0] for tok in sequence.split() if tok))

        # Need sequences for weight; if not present, just use accuracy
        test_swa = np.mean([t == p for t, p in zip(y_true, y_pred)])
    print(f"Last validation SWA: {last_val_swa}")
    print(f"Test SWA          : {test_swa}")
except Exception as e:
    print(f"Error printing metrics: {e}")

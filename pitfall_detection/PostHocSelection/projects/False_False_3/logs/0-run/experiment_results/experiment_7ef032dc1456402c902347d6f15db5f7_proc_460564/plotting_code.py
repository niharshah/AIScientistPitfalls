import matplotlib.pyplot as plt
import numpy as np
import os

# create / locate working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------
model_key, dset_key = "NoAuxVarLoss", "SPR_BENCH"
run = experiment_data.get(model_key, {}).get(dset_key, {})
loss_tr = run.get("losses", {}).get("train", [])
loss_val = run.get("losses", {}).get("val", [])
swa_val = run.get("metrics", {}).get("val", [])
y_pred = run.get("predictions", [])
y_true = run.get("ground_truth", [])


# Utility: accuracy
def simple_accuracy(y_true, y_pred):
    if not y_true:
        return None
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)


# ------------------------------------------------------
# PLOT 1: loss curves
try:
    if loss_tr and loss_val:
        epochs = np.arange(1, len(loss_tr) + 1)
        plt.figure()
        plt.plot(epochs, loss_tr, label="Train Loss")
        plt.plot(epochs, loss_val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
    else:
        print("Loss data missing – skipping loss curve.")
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# ------------------------------------------------------
# PLOT 2: validation SWA curve
try:
    if swa_val:
        epochs = np.arange(1, len(swa_val) + 1)
        plt.figure()
        plt.plot(epochs, swa_val, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation SWA per Epoch")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(fname)
    else:
        print("SWA data missing – skipping accuracy curve.")
except Exception as e:
    print(f"Error creating SWA curve: {e}")
finally:
    plt.close()

# ------------------------------------------------------
# PLOT 3: confusion matrix
try:
    if y_true and y_pred:
        labels = sorted(set(y_true) | set(y_pred))
        lab2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[lab2idx[t], lab2idx[p]] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth (rows), Right: Predicted (cols)"
        )
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    else:
        print("Prediction data missing – skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
finally:
    plt.close()

# ------------------------------------------------------
# Print overall accuracy
acc = simple_accuracy(y_true, y_pred)
if acc is not None:
    print(f"Test Classification Accuracy: {acc:.4f}")

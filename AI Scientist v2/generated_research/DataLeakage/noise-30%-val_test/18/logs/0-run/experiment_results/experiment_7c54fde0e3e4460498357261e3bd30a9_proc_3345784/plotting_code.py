import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Guard against missing data
if not experiment_data:
    print("No experiment data found; exiting.")
    exit()

run_key = list(experiment_data.keys())[0]  # 'NoCLS_MeanPooling'
ds_key = list(experiment_data[run_key].keys())[0]  # 'SPR_BENCH'
data = experiment_data[run_key][ds_key]

epochs = np.arange(1, len(data["losses"]["train"]) + 1)
train_loss = np.array(data["losses"]["train"])
val_loss = np.array(data["losses"]["val"])
train_mcc = np.array(data["metrics"]["train_MCC"])
val_mcc = np.array(data["metrics"]["val_MCC"])
preds = np.array(data["predictions"])
gts = np.array(data["ground_truth"])

# ---------- figure 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- figure 2: MCC curves ----------
try:
    plt.figure()
    plt.plot(epochs, train_mcc, label="Train MCC")
    plt.plot(epochs, val_mcc, label="Validation MCC")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews Corrcoef")
    plt.title("SPR_BENCH Training vs Validation MCC")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_MCC_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve: {e}")
    plt.close()

# ---------- figure 3: confusion matrix ----------
try:
    num_classes = len(np.unique(gts))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("SPR_BENCH Test Confusion Matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print final metrics ----------
best_val_mcc = max(val_mcc) if len(val_mcc) else None
test_mcc = np.nan
if cm.sum():
    # simple MCC recomputation
    tp = cm[1, 1] if cm.shape[0] > 1 else 0
    tn = cm[0, 0]
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    numerator = tp * tn - fp * fn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    test_mcc = numerator / denom if denom else np.nan

print(f"Best Val MCC = {best_val_mcc:.4f} | Test MCC (recomputed) = {test_mcc:.4f}")

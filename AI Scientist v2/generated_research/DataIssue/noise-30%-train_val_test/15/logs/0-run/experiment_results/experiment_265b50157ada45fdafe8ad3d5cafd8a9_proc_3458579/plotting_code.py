import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_key = "SPR_BENCH"
if data_key not in experiment_data:
    print(f"No data for {data_key} found – nothing to plot.")
    exit()

data = experiment_data[data_key]
loss_tr, loss_val = data["losses"]["train"], data["losses"]["val"]
f1_tr, f1_val = data["metrics"]["train"], data["metrics"]["val"]
preds = np.asarray(data["predictions"])
golds = np.asarray(data["ground_truth"])
epochs = np.arange(1, len(loss_tr) + 1)


# ---------------------------------------------------------------------
# Utility: macro-F1 without sklearn
def macro_f1(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return np.mean(f1s) if f1s else 0.0


# ---------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Macro-F1 curves
try:
    plt.figure()
    plt.plot(epochs, f1_tr, label="Train")
    plt.plot(epochs, f1_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH – Training vs Validation Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating f1 curve: {e}")
    plt.close()

# 3) Confusion matrix heat-map
try:
    num_classes = int(max(np.max(golds), np.max(preds))) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(golds, preds):
        cm[g, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH – Confusion Matrix (Test)")
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 4) Class distribution bar chart
try:
    gt_counts = np.bincount(golds, minlength=num_classes)
    pred_counts = np.bincount(preds, minlength=num_classes)
    ind = np.arange(num_classes)
    width = 0.35
    plt.figure()
    plt.bar(ind - width / 2, gt_counts, width, label="Ground Truth")
    plt.bar(ind + width / 2, pred_counts, width, label="Predictions")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("SPR_BENCH – Class Distribution (Test Set)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# Print evaluation metric
print(f"Test Macro-F1 (recomputed) = {macro_f1(golds, preds):.4f}")

import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths & data loading -----------------
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
if data_key not in experiment_data:
    raise RuntimeError(f"{data_key} not found in experiment_data.npy")

metrics = experiment_data[data_key]["metrics"]
preds_dev = np.array(experiment_data[data_key]["predictions"]["dev"])
gts_dev = np.array(experiment_data[data_key]["ground_truth"]["dev"])
preds_test = np.array(experiment_data[data_key]["predictions"]["test"])
gts_test = np.array(experiment_data[data_key]["ground_truth"]["test"])

epochs = np.arange(1, len(metrics["train_loss"]) + 1)

# ----------------- Figure 1: loss curves -----------------
try:
    plt.figure()
    plt.plot(epochs, metrics["train_loss"], label="Train")
    plt.plot(epochs, metrics["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------- Figure 2: validation metrics -----------------
try:
    plt.figure()
    plt.plot(epochs, metrics["val_swa"], label="SWA")
    plt.plot(epochs, metrics["val_cwa"], label="CWA")
    plt.plot(epochs, metrics["val_bps"], label="BPS")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Validation Metrics\nSWA, CWA, BPS over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()


# ----------------- helper: per-class accuracy -----------------
def per_class_acc(y_true, y_pred, num_classes):
    acc = np.zeros(num_classes)
    counts = np.zeros(num_classes)
    for t, p in zip(y_true, y_pred):
        counts[t] += 1
        if t == p:
            acc[t] += 1
    acc = np.divide(acc, counts, out=np.zeros_like(acc), where=counts > 0)
    return acc


num_classes = int(max(gts_test.max(), gts_dev.max()) + 1)

# ----------------- Figure 3: per-class accuracy -----------------
try:
    acc_dev = per_class_acc(gts_dev, preds_dev, num_classes)
    acc_test = per_class_acc(gts_test, preds_test, num_classes)
    x = np.arange(num_classes)
    width = 0.35
    plt.figure(figsize=(max(6, num_classes * 0.6), 4))
    plt.bar(x - width / 2, acc_dev, width, label="Dev")
    plt.bar(x + width / 2, acc_test, width, label="Test")
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Per-Class Accuracy\nDev vs Test")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_per_class_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating per-class accuracy plot: {e}")
    plt.close()

# ----------------- Figure 4: confusion matrix (test) -----------------
try:
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gts_test, preds_test):
        conf_mat[t, p] += 1
    plt.figure(figsize=(6, 5))
    im = plt.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH Confusion Matrix\nTest Split")
    fname = os.path.join(working_dir, "spr_bench_confusion_matrix_test.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ----------------- print summary metrics -----------------
overall_dev_acc = (preds_dev == gts_dev).mean() if gts_dev.size else 0.0
overall_test_acc = (preds_test == gts_test).mean() if gts_test.size else 0.0
print(f"Overall Dev Accuracy : {overall_dev_acc:.4f}")
print(f"Overall Test Accuracy: {overall_test_acc:.4f}")

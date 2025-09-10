import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data ---------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_key = "SPR_BENCH"
if data_key not in experiment_data:
    print(f"{data_key} not found in experiment_data")
    exit()

data = experiment_data[data_key]
loss_tr = data["losses"].get("train", [])
loss_val = data["losses"].get("val", [])
val_metrics = data["metrics"].get("val", [])
test_metrics = data["metrics"].get("test", {})
y_true = np.array(data.get("ground_truth", []))
y_pred = np.array(data.get("predictions", []))


# helper to extract metric curve -----------------------------------------
def metric_curve(metric_name):
    return [m.get(metric_name, np.nan) for m in val_metrics]


epochs = np.arange(1, len(loss_tr) + 1)

# 1. train / val loss -----------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"SPR_BENCH - Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2. validation metric curves --------------------------------------------
try:
    plt.figure()
    for m in ["swa", "cwa", "cwa2d"]:
        plt.plot(epochs, metric_curve(m), label=m.upper())
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH - Validation Weighted Accuracies")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metric_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()

# 3. bar chart of test metrics -------------------------------------------
try:
    plt.figure()
    names, vals = zip(*test_metrics.items()) if test_metrics else ([], [])
    plt.bar(names, vals, color="skyblue")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH - Test Weighted Accuracies")
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar plot: {e}")
    plt.close()

# 4. confusion matrix -----------------------------------------------------
try:
    if y_true.size and y_pred.size:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH - Confusion Matrix")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 5. class frequency histogram -------------------------------------------
try:
    if y_true.size and y_pred.size:
        plt.figure()
        bins = np.arange(int(max(num_classes, 1)) + 1) - 0.5
        plt.hist(y_true, bins=bins, alpha=0.6, label="Ground Truth")
        plt.hist(y_pred, bins=bins, alpha=0.6, label="Predicted")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title("SPR_BENCH - Class Distribution (True vs Predicted)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_class_histogram.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating histogram: {e}")
    plt.close()

# print test metrics ------------------------------------------------------
print("Test metrics:", test_metrics)

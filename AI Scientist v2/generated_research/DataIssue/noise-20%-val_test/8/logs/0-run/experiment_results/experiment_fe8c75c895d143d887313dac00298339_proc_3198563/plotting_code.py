import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------------------------------------------------------------------------
# Helper
def get_ds(name="SPR_BENCH"):
    return experiment_data.get(name, {})


ds = get_ds("SPR_BENCH")
if not ds:
    quit()  # nothing to plot

epochs = np.arange(len(ds["losses"]["train"]))

# -------------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, ds["losses"]["train"], marker="o", label="Train Loss")
    plt.plot(epochs, ds["losses"]["val"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 2) Accuracy curves
try:
    plt.figure()
    plt.plot(epochs, ds["metrics"]["train"], marker="o", label="Train Acc")
    plt.plot(epochs, ds["metrics"]["val"], marker="s", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 3) Confusion matrix on test split
try:
    from sklearn.metrics import confusion_matrix

    y_true = np.array(ds["ground_truth"])
    y_pred = np.array(ds["predictions"])
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 4) Test metrics bar chart (Accuracy vs SEFA)
try:
    plt.figure()
    bars = ["Test Acc", "SEFA"]
    values = [ds["metrics"]["test_acc"], ds["metrics"]["test_sefa"]]
    plt.bar(bars, values, color=["green", "orange"])
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Test Accuracy vs SEFA")
    for idx, val in enumerate(values):
        plt.text(idx, val + 0.02, f"{val:.2f}", ha="center")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

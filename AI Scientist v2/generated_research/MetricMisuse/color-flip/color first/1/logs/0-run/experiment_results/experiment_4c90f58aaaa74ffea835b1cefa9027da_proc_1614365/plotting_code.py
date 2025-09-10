import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tag = "no_color_embedding"
ds_name = "SPR"
exp = experiment_data.get(tag, {}).get(ds_name, {})


# helper to silently fetch dict keys
def g(path, default=None):
    cur = exp
    for p in path:
        cur = cur.get(p, {})
    return cur if cur else default


loss_train = g(["losses", "train"], [])
loss_val = g(["losses", "val"], [])
val_metrics = g(["metrics", "val"], [])
test_metrics = g(["metrics", "test"], {})
preds = exp.get("predictions", [])
gts = exp.get("ground_truth", [])

# ------------------ figure 1 : loss curves ------------------
try:
    plt.figure()
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR No-Color-Embedding: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves_no_color_embedding.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------ figure 2 : accuracy curves ------------------
try:
    if val_metrics:
        cwa = [m["cwa"] for m in val_metrics]
        swa = [m["swa"] for m in val_metrics]
        cva = [m["cva"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cva, label="CVA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR No-Color-Embedding: Validation Accuracies")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "SPR_accuracy_curves_no_color_embedding.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------ figure 3 : confusion matrix ------------------
try:
    if preds and gts:
        num_cls = max(max(preds), max(gts)) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for y, y_hat in zip(gts, preds):
            cm[y, y_hat] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted label")
        plt.ylabel("Ground truth label")
        plt.title(
            "SPR No-Color-Embedding: Confusion Matrix\n"
            "Rows: Ground Truth, Columns: Predictions"
        )
        plt.savefig(
            os.path.join(working_dir, "SPR_confusion_matrix_no_color_embedding.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------ print test metrics ------------------
if test_metrics:
    print("Test metrics:", test_metrics)

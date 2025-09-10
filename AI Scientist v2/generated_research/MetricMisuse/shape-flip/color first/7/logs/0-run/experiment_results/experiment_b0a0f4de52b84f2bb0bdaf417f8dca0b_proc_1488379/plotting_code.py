import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
data = experiment_data.get(ds_key, {})


# Helper to fetch safely
def g(path, default=None):
    cur = data
    for k in path:
        cur = cur.get(k, {})
    return cur if cur else default


# ----------------------------------------------------------------------
# 1) Loss curves -------------------------------------------------------
try:
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ----------------------------------------------------------------------
# 2) CoWA curves -------------------------------------------------------
try:
    train_cowa = data["metrics"]["train_CoWA"]
    val_cowa = data["metrics"]["val_CoWA"]
    epochs = range(1, len(train_cowa) + 1)

    plt.figure()
    plt.plot(epochs, train_cowa, label="Train")
    plt.plot(epochs, val_cowa, label="Validation")
    plt.title("SPR_BENCH: Training vs Validation CoWA")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_CoWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CoWA curve: {e}")
    plt.close()

# ----------------------------------------------------------------------
# 3) Confusion matrix (best-epoch) ------------------------------------
try:
    from itertools import product

    preds = np.array(data["predictions"])
    trues = np.array(data["ground_truth"])
    if preds.size and trues.size:
        num_classes = max(max(preds), max(trues)) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title("SPR_BENCH: Confusion Matrix (Val, best epoch)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        # annotate cells
        for i, j in product(range(num_classes), range(num_classes)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ----------------------------------------------------------------------
# 4) Epoch runtime bar chart ------------------------------------------
try:
    epoch_times = data["epoch_time"]
    if epoch_times:
        plt.figure()
        plt.bar(range(1, len(epoch_times) + 1), epoch_times)
        plt.title("SPR_BENCH: Epoch Runtime")
        plt.xlabel("Epoch")
        plt.ylabel("Seconds")
        fname = os.path.join(working_dir, "SPR_BENCH_epoch_time.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating epoch time plot: {e}")
    plt.close()

# ----------------------------------------------------------------------
# Print final test metrics --------------------------------------------
test_loss = data.get("test_loss", None)
test_cowa = data.get("test_CoWA", None)
if test_loss is not None and test_cowa is not None:
    print(f"Final Test Metrics -> Loss: {test_loss:.4f} | CoWA: {test_cowa:.4f}")

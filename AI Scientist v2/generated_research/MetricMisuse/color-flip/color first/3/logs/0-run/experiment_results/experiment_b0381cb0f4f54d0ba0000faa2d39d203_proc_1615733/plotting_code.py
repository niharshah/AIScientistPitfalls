import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# setup
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# convenient pointer
dname = ("no_early_stopping", "SPR_BENCH")
data = experiment_data.get(dname[0], {}).get(dname[1], {}) if experiment_data else {}

# ------------------------------------------------------------------ #
# 1. train / val loss curve
# ------------------------------------------------------------------ #
try:
    train = np.array(data["losses"]["train"])
    val = np.array(data["losses"]["val"])
    epochs_t, loss_t = train[:, 0], train[:, 1]
    epochs_v, loss_v = val[:, 0], val[:, 1]

    plt.figure()
    plt.plot(epochs_t, loss_t, label="Train")
    plt.plot(epochs_v, loss_v, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curve\nTrain vs. Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve_no_early_stopping.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2. validation metrics over epochs
# ------------------------------------------------------------------ #
try:
    metrics = np.array(data["metrics"]["val"])
    epochs = metrics[:, 0]
    cwa, swa, hcs, snwa = metrics[:, 1], metrics[:, 2], metrics[:, 3], metrics[:, 4]

    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, hcs, label="HCSA")
    plt.plot(epochs, snwa, label="SNWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Validation Metrics Over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics_no_early_stopping.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3. confusion matrix on test split
# ------------------------------------------------------------------ #
try:
    preds = np.array(data["predictions"]["test"])
    gts = np.array(data["ground_truth"]["test"])
    n_cls = int(max(preds.max(), gts.max())) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("SPR_BENCH Confusion Matrix – Test Split")
    plt.tight_layout()
    fname = os.path.join(
        working_dir, "SPR_BENCH_confusion_matrix_no_early_stopping.png"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

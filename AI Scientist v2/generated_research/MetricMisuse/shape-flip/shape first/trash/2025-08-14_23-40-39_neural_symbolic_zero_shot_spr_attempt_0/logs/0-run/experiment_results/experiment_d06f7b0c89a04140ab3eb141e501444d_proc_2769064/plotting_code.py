import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data["SPR_BENCH"]
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    metrics = data["metrics"]["val"]
    epochs = np.arange(1, len(train_loss) + 1)

    swa = [m["SWA"] for m in metrics]
    cwa = [m["CWA"] for m in metrics]
    bps = [m["BPS"] for m in metrics]

    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    unique_labels = sorted(list(set(gts) | set(preds)))
    n_cls = len(unique_labels)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for gt, pr in zip(gts, preds):
        cm[unique_labels.index(gt), unique_labels.index(pr)] += 1

    # ------------------------------------------------------------------
    # Plot 1: Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Metric curves
    try:
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, bps, label="BPS")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation Metrics Across Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Confusion matrix
    try:
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(n_cls), unique_labels)
        plt.yticks(range(n_cls), unique_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Print final metrics
    if metrics:
        print(
            "Final Epoch Metrics - SWA: {:.3f}, CWA: {:.3f}, BPS: {:.3f}".format(
                swa[-1], cwa[-1], bps[-1]
            )
        )

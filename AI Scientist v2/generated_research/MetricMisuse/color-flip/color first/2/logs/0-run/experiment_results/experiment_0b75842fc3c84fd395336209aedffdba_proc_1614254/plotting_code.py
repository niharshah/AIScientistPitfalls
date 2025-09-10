import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["RandomCluster"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = {}

# 1) Loss curves
try:
    train_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    epochs = range(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Validation metric curves
try:
    val_metrics = run["metrics"]["val"]
    cwa = [m["CWA"] for m in val_metrics]
    swa = [m["SWA"] for m in val_metrics]
    gcwa = [m["GCWA"] for m in val_metrics]
    epochs = range(1, len(cwa) + 1)
    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, gcwa, label="GCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Validation Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics curve: {e}")
    plt.close()

# 3) Confusion matrix on test set
try:
    preds = np.array(run["predictions"])
    tgts = np.array(run["ground_truth"])
    if preds.size and tgts.size:
        num_classes = max(preds.max(), tgts.max()) + 1
        cm = np.zeros((num_classes, num_classes), int)
        for t, p in zip(tgts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

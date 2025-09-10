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


# Helper to fetch subtree safely
def get_spr(exp_dict):
    return exp_dict.get("monohead", {}).get("SPR", {})


spr = get_spr(experiment_data)

# 1) Loss curves -----------------------------------------------------------------
try:
    losses = spr.get("losses", {})
    tr_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if tr_loss and val_loss:
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.title(
            "SPR Dataset – Training vs Validation Loss\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Metric curves ---------------------------------------------------------------
try:
    val_metrics = spr.get("metrics", {}).get("val", [])
    if val_metrics:
        cwa = [m["cwa"] for m in val_metrics]
        swa = [m["swa"] for m in val_metrics]
        cva = [m["cva"] for m in val_metrics]
        epochs = np.arange(1, len(cwa) + 1)
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cva, label="CVA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR Dataset – Weighted Accuracy Metrics (Validation)\nLeft: CWA, Middle: SWA, Right: CVA"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_metric_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# 3) Confusion matrix ------------------------------------------------------------
try:
    preds = np.array(spr.get("predictions", []))
    gts = np.array(spr.get("ground_truth", []))
    if preds.size and gts.size:
        classes = sorted(set(gts) | set(preds))
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            matrix[t, p] += 1
        plt.figure()
        im = plt.imshow(matrix, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(
                    j,
                    i,
                    matrix[i, j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
        plt.title(
            "SPR Dataset – Confusion Matrix (Test)\nLeft: Ground Truth, Right: Predictions"
        )
        fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    # ---- unpack helpers ----
    def unpack(pairs):  # list of (ts,val) or None
        return [v for ts, v in pairs if v is not None] if pairs else []

    epochs = range(1, len(data["losses"]["train"]) + 1)
    train_loss = unpack(data["losses"]["train"])
    val_loss = unpack(data["losses"]["val"])
    val_dwa = unpack(data["metrics"]["val"])
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    classes = sorted(set(gts) | set(preds))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(gts, preds):
        cm[classes.index(t), classes.index(p)] += 1

    # -------------- plots --------------
    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, "-o", label="Train Loss")
        plt.plot(epochs, val_loss, "-o", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. DWA curve
    try:
        plt.figure()
        plt.plot(epochs, val_dwa, "-o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Dual-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation Dual-Weighted Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_DWA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating DWA plot: {e}")
        plt.close()

    # 3. Confusion matrix
    try:
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(
            "SPR_BENCH: Confusion Matrix\nLeft axis: Ground Truth, Bottom axis: Predicted"
        )
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        # annotate cells
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

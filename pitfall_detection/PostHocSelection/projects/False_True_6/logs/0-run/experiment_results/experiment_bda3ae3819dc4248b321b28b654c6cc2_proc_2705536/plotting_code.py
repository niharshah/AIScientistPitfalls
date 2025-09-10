import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    rec = experiment_data["CLS_Pooling"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    rec = None

# ----------------- plot 1: loss curves -----------------
try:
    if rec is None:
        raise ValueError("Experiment record missing")
    train_loss = rec["losses"]["train"]
    val_loss = rec["losses"]["val"]
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------- plot 2: validation SWA -----------------
try:
    if rec is None:
        raise ValueError("Experiment record missing")
    val_swa = rec["SWA"]["val"]
    epochs = np.arange(1, len(val_swa) + 1)

    plt.figure()
    plt.plot(epochs, val_swa, marker="o", label="Val SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH – Validation SWA Curve")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_validation_SWA.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ----------------- plot 3: confusion matrix -----------------
try:
    if rec is None:
        raise ValueError("Experiment record missing")
    y_pred = np.array(rec["predictions"])
    y_true = np.array(rec["ground_truth"])
    num_labels = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(
        "SPR_BENCH – Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
    )
    # annotate cells
    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

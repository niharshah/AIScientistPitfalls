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

# Extract data
try:
    d = experiment_data["Frozen-Random-Embeddings"]["SPR_BENCH"]
    epochs = d["epochs"]
    train_losses = d["losses"]["train"]
    val_losses = d["losses"]["val"]
    val_swa = d["metrics"]["val"]
    preds_all = d["predictions"]
    gts_all = d["ground_truth"]
except Exception as e:
    print(f"Error extracting data: {e}")
    d, epochs, train_losses, val_losses, val_swa, preds_all, gts_all = (
        {},
        [],
        [],
        [],
        [],
        [],
        [],
    )

# Figure 1: loss curves
try:
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Figure 2: validation SWA curve
try:
    plt.figure()
    plt.plot(epochs, val_swa, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy (SWA)")
    plt.title("SPR_BENCH – Validation SWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_SWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# Figure 3: confusion matrix at best epoch
try:
    best_idx = int(np.argmin(val_losses))
    y_true = np.array(gts_all[best_idx])
    y_pred = np.array(preds_all[best_idx])
    n_cls = len(np.unique(np.concatenate([y_true, y_pred])))
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"SPR_BENCH – Confusion Matrix (Best Epoch {epochs[best_idx]})")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fname = f"SPR_BENCH_confusion_matrix_epoch{epochs[best_idx]}.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- setup & data loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


dataset = "SPR_BENCH"
logs = experiment_data.get(dataset, {})

# ---------------- 1. Loss curves -----------------------
try:
    train_loss = safe_get(logs, "losses", "train", default=[])
    val_loss = safe_get(logs, "losses", "val", default=[])
    if train_loss and val_loss:
        epochs = np.arange(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset} – Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------------- 2. CCWA curve ------------------------
try:
    val_ccwa = safe_get(logs, "metrics", "val_CCWA", default=[])
    if val_ccwa:
        epochs = np.arange(1, len(val_ccwa) + 1)
        plt.figure()
        plt.plot(epochs, val_ccwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.title(f"{dataset} – Validation CCWA")
        fname = os.path.join(working_dir, f"{dataset}_val_CCWA.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CCWA curve: {e}")
    plt.close()

# ---------------- 3. Confusion matrix ------------------
try:
    preds = safe_get(logs, "predictions", default=[])
    gts = safe_get(logs, "ground_truth", default=[])
    if preds and gts:
        y_pred = np.array(preds[-1])
        y_true = np.array(gts[-1])
        num_cls = int(max(y_true.max(), y_pred.max()) + 1)
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{dataset} – Confusion Matrix (Final Epoch)")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, f"{dataset}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- 4. Print final metric ----------------
final_ccwa = val_ccwa[-1] if val_ccwa else None
print(f"Final Validation CCWA: {final_ccwa}")
print("Plotting complete; figures saved to", working_dir)

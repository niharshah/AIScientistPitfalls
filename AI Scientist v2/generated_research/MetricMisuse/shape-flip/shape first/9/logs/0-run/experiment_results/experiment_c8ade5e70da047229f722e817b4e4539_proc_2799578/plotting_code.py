import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = {}


# ----------------------------- helpers ------------------------------------
def safe_len(x):
    return len(x) if isinstance(x, (list, tuple)) else 0


# ----------------------------- Plot 1 -------------------------------------
try:
    tr_loss, va_loss = run["losses"]["train"], run["losses"]["val"]
    if safe_len(tr_loss) and safe_len(va_loss):
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, va_loss, linestyle="--", label="Validation")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    else:
        print("Loss data missing – skipping loss plot")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------------------- Plot 2 -------------------------------------
try:
    val_swa = run["metrics"]["val"]
    if safe_len(val_swa):
        epochs = np.arange(1, len(val_swa) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_swa, marker="o")
        plt.title("SPR_BENCH Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_SWA_curves.png"))
    else:
        print("Validation SWA data missing – skipping SWA plot")
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve plot: {e}")
    plt.close()

# ----------------------------- Plot 3 -------------------------------------
try:
    preds, gts = run["predictions"], run["ground_truth"]
    if safe_len(preds) and safe_len(gts):
        classes = sorted(set(gts) | set(preds))
        n_cls = len(classes)
        cls2idx = {c: i for i, c in enumerate(classes)}
        conf = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            conf[cls2idx[t], cls2idx[p]] += 1

        plt.figure(figsize=(4, 4))
        im = plt.imshow(conf, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.title("SPR_BENCH Confusion Matrix\nLeft: True, Right: Predicted")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(range(n_cls), classes)
        plt.yticks(range(n_cls), classes)
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, conf[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    else:
        print("Prediction data missing – skipping confusion matrix")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ----------------------------- Metrics print ------------------------------
try:
    test_swa = run["metrics"]["test"]
    if test_swa is not None:
        print(f"Final Test Shape-Weighted Accuracy: {test_swa:.4f}")
    if safe_len(preds) and safe_len(gts):
        per_cls_acc = {
            c: (np.array(preds)[np.array(gts) == c] == c).mean() for c in classes
        }
        print("Per-class accuracy:", per_cls_acc)
except Exception as e:
    print(f"Error printing metrics: {e}")

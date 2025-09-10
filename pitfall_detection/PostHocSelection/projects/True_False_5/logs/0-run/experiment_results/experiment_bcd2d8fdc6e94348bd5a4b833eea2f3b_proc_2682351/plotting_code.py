import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("SYM_ONLY", {}).get("SPR_BENCH", None)
if exp is None:
    print("SPR_BENCH record not found in experiment_data.npy")
    exit()


# ---------- helpers ----------
def safe_list(x):
    return x if isinstance(x, (list, np.ndarray)) else []


train_loss = safe_list(exp["losses"].get("train", []))
val_loss = safe_list(exp["losses"].get("val", []))
train_swa = safe_list(exp["metrics"].get("train_swa", []))
val_swa = safe_list(exp["metrics"].get("val_swa", []))
preds = np.array(exp.get("predictions", []))
gts = np.array(exp.get("ground_truth", []))

# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    epochs = np.arange(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- plot 2: SWA curves ----------
try:
    plt.figure()
    epochs = np.arange(1, len(train_swa) + 1)
    plt.plot(epochs, train_swa, label="Train SWA")
    plt.plot(epochs, val_swa, label="Validation SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH: Training vs Validation SWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curves: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    if preds.size and gts.size:
        num_classes = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted label")
        plt.ylabel("Ground truth label")
        plt.title("SPR_BENCH: Confusion Matrix (GT rows, Pred cols)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    else:
        print("Predictions / ground-truth arrays empty; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print metrics ----------
try:
    final_train_swa = train_swa[-1] if train_swa else float("nan")
    final_val_swa = val_swa[-1] if val_swa else float("nan")
    test_acc = (preds == gts).mean() if preds.size and gts.size else float("nan")
    print(f"Final Train SWA: {final_train_swa:.4f}")
    print(f"Final Val   SWA: {final_val_swa:.4f}")
    print(f"Test Accuracy:   {test_acc:.4f}")
except Exception as e:
    print(f"Error printing metrics: {e}")

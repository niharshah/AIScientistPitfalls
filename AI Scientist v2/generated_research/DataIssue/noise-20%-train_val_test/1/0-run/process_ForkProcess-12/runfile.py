import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------ set up and load ---------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

if "SPR_BENCH" not in experiment_data:
    print("No SPR_BENCH data found; nothing to plot.")
    exit()

spr = experiment_data["SPR_BENCH"]


# Helper to safely fetch arrays
def _np(arr_name, d, fallback=[]):
    return np.asarray(d.get(arr_name, fallback), dtype=float)


train_loss = _np("train", spr["losses"])
val_loss = _np("val", spr["losses"])
train_acc = _np("train_acc", spr["metrics"])
val_acc = _np("val_acc", spr["metrics"])
train_f1 = _np("train_f1", spr["metrics"])
val_f1 = _np("val_f1", spr["metrics"])
epochs = np.arange(1, len(train_loss) + 1)


# ------------------------ plotting helpers ---------------------------------- #
def _safe_save(fig_name):
    return os.path.join(working_dir, f"spr_bench_{fig_name}.png")


# 1. Loss Curves
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("SPR_BENCH Loss Curves (Sequence Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(_safe_save("loss_curves"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. Accuracy Curves
try:
    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.title("SPR_BENCH Accuracy Curves (Sequence Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(_safe_save("accuracy_curves"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 3. Macro-F1 Curves
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.title("SPR_BENCH Macro-F1 Curves (Sequence Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.savefig(_safe_save("f1_curves"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 4. Confusion Matrix on Test Set
try:
    preds = np.asarray(spr.get("predictions", []), dtype=int)
    gts = np.asarray(spr.get("ground_truth", []), dtype=int)
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix (Test Set)")
        plt.xlabel("Predicted Label")
        plt.ylabel("Ground Truth Label")
        ticks = np.arange(num_classes)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.savefig(_safe_save("confusion_matrix"))
        plt.close()

        test_acc = (preds == gts).mean()
        # Avoid div/0 if single class missing
        f1_per_class = []
        for c in range(num_classes):
            tp = ((preds == c) & (gts == c)).sum()
            fp = ((preds == c) & (gts != c)).sum()
            fn = ((preds != c) & (gts == c)).sum()
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            f1_per_class.append(f1)
        macro_f1 = float(np.mean(f1_per_class))
        print(f"Test accuracy: {test_acc*100:.2f}% | Test macro-F1: {macro_f1:.4f}")
    else:
        print("Predictions or ground-truth not found; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

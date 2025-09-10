import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("SPR_BENCH", {})
metrics = ed.get("metrics", {})
epochs = ed.get("epochs", [])
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# ------------------ plot 1: loss curves ------------------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_loss", []), label="Train Loss")
    plt.plot(epochs, metrics.get("val_loss", []), label="Validation Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------ plot 2: validation F1 ------------------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("val_f1", []), marker="o", color="green")
    plt.title("SPR_BENCH Validation Macro-F1 Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_f1_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ------------------ plot 3: confusion matrix ------------------
from sklearn.metrics import confusion_matrix, f1_score

if preds.size and gts.size:
    try:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

# ------------------ plot 4: class distribution ------------------
if preds.size and gts.size:
    try:
        classes = np.arange(max(gts.max(), preds.max()) + 1)
        gt_counts = np.array([np.sum(gts == c) for c in classes])
        pred_counts = np.array([np.sum(preds == c) for c in classes])
        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predicted")
        plt.title("SPR_BENCH Class Distribution\nLeft: Ground Truth, Right: Predicted")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(x, classes)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_class_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating class distribution plot: {e}")
        plt.close()

# ------------------ print final metric ------------------
if preds.size and gts.size:
    final_f1 = f1_score(gts, preds, average="macro")
    print(f"Final Test Macro-F1: {final_f1:.4f}")

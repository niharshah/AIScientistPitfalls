import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix

# ---------- load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

test_scores = {}

for dset, rec in experiment_data.items():
    epochs = np.asarray(rec.get("epochs", []))
    metrics = rec.get("metrics", {})
    losses = rec.get("losses", {})
    preds = np.asarray(rec.get("predictions", []))
    gts = np.asarray(rec.get("ground_truth", []))

    # ---------- plot 1: F1 curves ----------
    try:
        if len(epochs) and metrics.get("train_f1") and metrics.get("val_f1"):
            plt.figure()
            plt.plot(epochs, metrics["train_f1"], label="Train")
            plt.plot(epochs, metrics["val_f1"], linestyle="--", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dset}: Train vs Validation Macro-F1\nLeft: Train, Right: Val")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_f1_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {dset}: {e}")
        plt.close()

    # ---------- plot 2: Loss curves ----------
    try:
        if len(epochs) and losses.get("train") and losses.get("val"):
            plt.figure()
            plt.plot(epochs, losses["train"], label="Train")
            plt.plot(epochs, losses["val"], linestyle="--", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset}: Train vs Validation Loss\nLeft: Train, Right: Val")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ---------- plot 3: Confusion matrix ----------
    try:
        if preds.size and gts.size and preds.shape == gts.shape:
            cm = confusion_matrix(gts, preds)
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title(f"{dset}: Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

    # ---------- compute / collect test F1 ----------
    try:
        test_f1 = metrics.get("test_f1")
        if test_f1 is None and preds.size and gts.size:
            test_f1 = f1_score(gts, preds, average="macro")
        if test_f1 is not None:
            test_scores[dset] = float(test_f1)
    except Exception as e:
        print(f"Error computing test F1 for {dset}: {e}")

# ---------- bar chart across datasets ----------
try:
    if test_scores:
        plt.figure()
        keys = list(test_scores.keys())
        vals = [test_scores[k] for k in keys]
        plt.bar(keys, vals, color="skyblue")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.ylabel("Macro-F1")
        plt.title("Test Macro-F1 Comparison Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "datasets_test_f1_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating comparison bar chart: {e}")
    plt.close()

# ---------- print summary ----------
print("=== Test Macro-F1 Scores ===")
for k, v in test_scores.items():
    print(f"{k}: {v:.4f}")

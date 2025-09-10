import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------- LOAD DATA -----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Assume single dataset key
if experiment_data:
    dname = list(experiment_data.keys())[0]
    data = experiment_data[dname]
    epochs = data["epochs"]
    train_loss, val_loss = data["losses"]["train"], data["losses"]["val"]
    train_f1, val_f1 = data["metrics"]["train_f1"], data["metrics"]["val_f1"]
    preds, gts = np.array(data["predictions"]), np.array(data["ground_truth"])

    # ----------------------- PLOT 1: LOSS CURVE -----------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ----------------------- PLOT 2: F1 CURVE -----------------------
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train Macro F1")
        plt.plot(epochs, val_f1, label="Val Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(f"{dname} Macro-F1 Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_f1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ----------------------- PLOT 3: CONFUSION MATRIX -----------------------
    try:
        num_classes = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{dname} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ----------------------- PLOT 4: LABEL DISTRIBUTION -----------------------
    try:
        labels = np.arange(int(max(gts.max(), preds.max()) + 1))
        gt_counts = np.bincount(gts, minlength=len(labels))
        pred_counts = np.bincount(preds, minlength=len(labels))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        axes[0].bar(labels, gt_counts)
        axes[0].set_title("Left: Ground Truth")
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")

        axes[1].bar(labels, pred_counts, color="orange")
        axes[1].set_title("Right: Generated Samples")
        axes[1].set_xlabel("Class")

        fig.suptitle(f"{dname} Label Distribution Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_label_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating label distribution plot: {e}")
        plt.close()

    # ----------------------- PRINT METRIC -----------------------
    print(f"Final validation Macro-F1 (from file): {val_f1[-1]:.4f}")

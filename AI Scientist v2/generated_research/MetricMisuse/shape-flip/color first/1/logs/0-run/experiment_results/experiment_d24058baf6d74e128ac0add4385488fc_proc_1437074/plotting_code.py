import matplotlib.pyplot as plt
import numpy as np
import os

# required working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick guard
if "SPR" not in experiment_data:
    print("No SPR experiment data found, exiting.")
    exit()

spr = experiment_data["SPR"]
train_losses = spr["losses"]["train"]
val_losses = spr["losses"]["val"]
val_metrics = spr["metrics"]["val"]  # list of dicts

epochs = list(range(1, len(train_losses) + 1))

# gather metric curves
acc_curve = [m["acc"] for m in val_metrics]
cwa_curve = [m["cwa"] for m in val_metrics]
swa_curve = [m["swa"] for m in val_metrics]
caa_curve = [m["caa"] for m in val_metrics]

preds = np.array(spr.get("predictions", []))
gts = np.array(spr.get("ground_truth", []))


# ------------- helper to close safely -------------
def close_fig():
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()


# ------------- Plot 1: Loss curves -------------
try:
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Dataset – Training vs. Validation Loss")
    plt.legend()
    fname = "SPR_loss_curve.png"
    close_fig()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------- Plot 2: Validation Accuracy -------------
try:
    plt.figure()
    plt.plot(epochs, acc_curve, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR Dataset – Validation Accuracy per Epoch")
    fname = "SPR_val_accuracy.png"
    close_fig()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# ------------- Plot 3: Weighted Accuracies -------------
try:
    plt.figure()
    plt.plot(epochs, cwa_curve, label="CWA")
    plt.plot(epochs, swa_curve, label="SWA")
    plt.plot(epochs, caa_curve, label="CAA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR Dataset – Weighted Accuracies per Epoch")
    plt.legend()
    fname = "SPR_weighted_accuracies.png"
    close_fig()
except Exception as e:
    print(f"Error creating weighted accuracy curves: {e}")
    plt.close()

# ------------- Plot 4: Confusion Matrix -------------
try:
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("SPR Dataset – Confusion Matrix (Dev Set)")
        fname = "SPR_confusion_matrix.png"
        close_fig()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------- Print final metrics -------------
if preds.size and gts.size:
    final_idx = -1
    print("Final Validation Metrics:")
    print(f"  Accuracy: {acc_curve[final_idx]:.3f}")
    print(f"  Color-Weighted Accuracy: {cwa_curve[final_idx]:.3f}")
    print(f"  Shape-Weighted Accuracy: {swa_curve[final_idx]:.3f}")
    print(f"  Complexity-Adjusted Accuracy: {caa_curve[final_idx]:.3f}")

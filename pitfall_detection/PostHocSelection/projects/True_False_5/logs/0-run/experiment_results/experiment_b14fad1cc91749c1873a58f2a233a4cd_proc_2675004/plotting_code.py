import matplotlib.pyplot as plt
import numpy as np
import os

# -------- load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

name = "SPR_BENCH"
data = experiment_data.get(name, {})
epochs = data.get("epochs", [])
train_loss = data.get("losses", {}).get("train", [])
val_loss = data.get("losses", {}).get("val", [])
val_rcwa = data.get("metrics", {}).get("val", [])
test_preds = np.array(data.get("predictions", []))
test_truth = np.array(data.get("ground_truth", []))
test_rcwa = data.get("metrics", {}).get("test_rcwa", None)

# 1) Loss curve --------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) RCWA curve --------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, val_rcwa, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("RCWA")
    plt.title("SPR_BENCH: Validation RCWA over Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_RCWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RCWA curve: {e}")
    plt.close()

# 3) Confusion matrix --------------------------------------------------------
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(test_truth, test_preds, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("SPR_BENCH: Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 4) Prediction distribution -------------------------------------------------
try:
    plt.figure()
    unique, counts_pred = np.unique(test_preds, return_counts=True)
    unique_t, counts_true = np.unique(test_truth, return_counts=True)
    width = 0.35
    idx = np.arange(len(unique))
    plt.bar(idx - width / 2, counts_true, width, label="Ground Truth")
    plt.bar(idx + width / 2, counts_pred, width, label="Predictions")
    plt.xticks(idx, [f"Class {u}" for u in unique])
    plt.ylabel("Count")
    plt.title(
        "SPR_BENCH: Test Label Distribution\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_label_distribution.png"))
    plt.close()
except Exception as e:
    print(f"Error creating label distribution plot: {e}")
    plt.close()

# -------- print metric ----------
if test_rcwa is not None:
    print(f"Loaded Test RCWA: {test_rcwa:.4f}")

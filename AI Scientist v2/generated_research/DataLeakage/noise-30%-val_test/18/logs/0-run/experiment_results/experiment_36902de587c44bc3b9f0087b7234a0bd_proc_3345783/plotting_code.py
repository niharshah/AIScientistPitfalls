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

exp = experiment_data.get("RemovePositionalEmbedding", {}).get("SPR_BENCH", {})
train_loss = exp.get("losses", {}).get("train", [])
val_loss = exp.get("losses", {}).get("val", [])
train_mcc = exp.get("metrics", {}).get("train_MCC", [])
val_mcc = exp.get("metrics", {}).get("val_MCC", [])
preds = np.array(exp.get("predictions", []))
gts = np.array(exp.get("ground_truth", []))

epochs = np.arange(1, len(train_loss) + 1)

# --------- Loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# --------- MCC curves ------------
try:
    plt.figure()
    plt.plot(epochs, train_mcc, label="Train MCC")
    plt.plot(epochs, val_mcc, label="Val MCC")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews CorrCoef")
    plt.title("SPR_BENCH: Training vs Validation MCC")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve plot: {e}")
    plt.close()

# --------- Confusion matrix -------
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(gts, preds) if preds.size else np.zeros((2, 2))
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "SPR_BENCH: Test Confusion Matrix\nLeft: Ground Truth, Right: Predicted Labels"
    )
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

print("Plots saved to", working_dir)

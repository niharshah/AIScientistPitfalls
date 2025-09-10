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

dataset = "SPR_BENCH"
data = experiment_data.get(dataset, {})
metrics = data.get("metrics", {})
losses = data.get("losses", {})
preds = np.array(data.get("predictions", []))
gts = np.array(data.get("ground_truth", []))

epochs = range(1, len(metrics.get("train_acc", [])) + 1)

# ---------- plot 1: accuracy curves ----------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_acc", []), label="Train Acc")
    plt.plot(epochs, metrics.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset} – Train vs Val Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- plot 2: training loss ----------
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset} – Training Loss Curve")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 3: validation URA ----------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("val_ura", []), label="Val URA", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("URA")
    plt.title(f"{dataset} – Validation URA over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_val_ura_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating URA plot: {e}")
    plt.close()

# ---------- plot 4: confusion matrix ----------
try:
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(f"{dataset} – Confusion Matrix (Test)")
        fname = os.path.join(working_dir, f"{dataset}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print evaluation metrics ----------
if preds.size and gts.size:
    test_acc = (preds == gts).mean()
    print(f"Test Accuracy: {test_acc:.3f}  |  Epochs plotted: {len(list(epochs))}")

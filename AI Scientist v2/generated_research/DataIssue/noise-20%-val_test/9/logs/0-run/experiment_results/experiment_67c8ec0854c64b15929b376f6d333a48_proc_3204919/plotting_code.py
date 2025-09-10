import matplotlib.pyplot as plt
import numpy as np
import os

# ----- paths -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("SPR_BENCH", {})


# Helper to safely fetch arrays
def get(arr, key):
    return arr.get(key, [])


metrics = data.get("metrics", {})
losses = data.get("losses", {})

# Plot 1: Accuracy curves
try:
    train_acc = np.array(get(metrics, "train_acc"))
    val_acc = np.array(get(metrics, "val_acc"))
    if train_acc.size and val_acc.size:
        plt.figure()
        epochs = np.arange(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 2: Loss curves
try:
    train_loss = np.array(get(losses, "train"))
    val_loss = np.array(get(losses, "val"))
    if train_loss.size and val_loss.size:
        plt.figure()
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 3: RBA vs Validation Accuracy
try:
    rba = np.array(get(metrics, "RBA"))
    val_acc = np.array(get(metrics, "val_acc"))
    if rba.size and val_acc.size:
        plt.figure()
        epochs = np.arange(1, len(rba) + 1)
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.plot(epochs, rba, label="RBA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Accuracy vs Rule-Based Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rba_vs_val.png"))
        plt.close()
except Exception as e:
    print(f"Error creating RBA plot: {e}")
    plt.close()

# Plot 4: Confusion Matrix
try:
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    if preds.size and gts.size and preds.shape == gts.shape:
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ----- evaluation metric -----
if "predictions" in data and "ground_truth" in data:
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    if preds.size and preds.shape == gts.shape:
        test_acc = (preds == gts).mean()
        print(f"Test Accuracy: {test_acc:.3f}")

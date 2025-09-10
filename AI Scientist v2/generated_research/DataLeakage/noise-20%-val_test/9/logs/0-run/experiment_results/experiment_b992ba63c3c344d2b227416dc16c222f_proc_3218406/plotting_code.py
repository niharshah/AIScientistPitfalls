import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to fetch nested dict safely
def get(path, default=None):
    obj = experiment_data
    for key in path:
        if key not in obj:
            return default
        obj = obj[key]
    return obj


# Extract data for plotting
train_acc = get(["RuleFreeCNN", "SPR_BENCH", "metrics", "train"], [])
val_acc = get(["RuleFreeCNN", "SPR_BENCH", "metrics", "val"], [])
train_loss = get(["RuleFreeCNN", "SPR_BENCH", "losses", "train"], [])
val_loss = get(["RuleFreeCNN", "SPR_BENCH", "losses", "val"], [])
preds = get(["RuleFreeCNN", "SPR_BENCH", "predictions"], np.array([]))
gts = get(["RuleFreeCNN", "SPR_BENCH", "ground_truth"], np.array([]))

# 1) Accuracy curve
try:
    if len(train_acc) and len(val_acc):
        epochs = np.arange(1, len(train_acc) + 1)
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("RuleFreeCNN on SPR_BENCH - Accuracy vs Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve_RuleFreeCNN.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 2) Loss curve
try:
    if len(train_loss) and len(val_loss):
        epochs = np.arange(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("RuleFreeCNN on SPR_BENCH - Loss vs Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve_RuleFreeCNN.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 3) Confusion matrix
try:
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("RuleFreeCNN on SPR_BENCH - Confusion Matrix (Test Set)")
        plt.xticks(np.arange(num_classes))
        plt.yticks(np.arange(num_classes))
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_RuleFreeCNN.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()

        # Print overall accuracy
        accuracy = np.trace(cm) / cm.sum() if cm.sum() else 0.0
        print(f"Test accuracy (recomputed): {accuracy:.3f}")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

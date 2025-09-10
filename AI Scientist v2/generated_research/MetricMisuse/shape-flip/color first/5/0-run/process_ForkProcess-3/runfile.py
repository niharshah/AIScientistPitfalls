import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    def unique_shapes(seq):
        return len({tok[0] for tok in seq.strip().split() if tok})

    def unique_colors(seq):
        return len({tok[1] for tok in seq.strip().split() if len(tok) > 1})

    weights = [unique_shapes(s) * unique_colors(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / sum(weights) if sum(weights) else 0.0


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
data = experiment_data.get(ds_name, {})

# ---------------- Loss curves ----------------
try:
    epochs = range(1, len(data["losses"]["train"]) + 1)
    plt.figure()
    plt.plot(epochs, data["losses"]["train"], label="Train Loss")
    plt.plot(epochs, data["losses"]["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------- CpxWA curves ---------------
try:
    plt.figure()
    plt.plot(epochs, data["metrics"]["train_CpxWA"], label="Train CpxWA")
    plt.plot(epochs, data["metrics"]["val_CpxWA"], label="Val CpxWA")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH CpxWA Curves\nLeft: Train, Right: Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_cpxwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA plot: {e}")
    plt.close()

# ---------------- Confusion matrix -----------
try:
    y_true = np.array(data["ground_truth"])
    y_pred = np.array(data["predictions"])
    num_classes = max(y_true.max(), y_pred.max()) + 1 if y_true.size else 0
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted Samples"
    )
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------- Print evaluation metrics ---
try:
    seqs_for_test = data.get("seqs_test", [])  # may not exist
    if not seqs_for_test:  # fall back if sequences not stored
        seqs_for_test = [""] * len(y_true)
    cpxwa_test = complexity_weighted_accuracy(seqs_for_test, y_true, y_pred)
    simple_acc = (y_true == y_pred).mean() if y_true.size else 0.0
    print(f"Test Accuracy: {simple_acc:.4f}")
    print(f"Test Complexity-Weighted Accuracy: {cpxwa_test:.4f}")
except Exception as e:
    print(f"Error computing evaluation metrics: {e}")

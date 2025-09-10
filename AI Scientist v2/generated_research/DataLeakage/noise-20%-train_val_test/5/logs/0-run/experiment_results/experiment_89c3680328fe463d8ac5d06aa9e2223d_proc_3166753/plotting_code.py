import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# ------------------------------------------------------------------ #
# 1. F1 curves
try:
    train_f1 = data["metrics"]["train_f1"]
    val_f1 = data["metrics"]["val_f1"]
    if train_f1 and val_f1:
        epochs = np.arange(1, len(train_f1) + 1)
        plt.figure()
        plt.plot(epochs, train_f1, marker="o", label="Train F1")
        plt.plot(epochs, val_f1, marker="x", label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2. Loss curves
try:
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    if train_loss and val_loss:
        epochs = np.arange(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, marker="o", label="Train Loss")
        plt.plot(epochs, val_loss, marker="x", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3. Confusion matrix (test)
try:
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    if preds.size and gts.size and preds.shape == gts.shape:
        num_classes = int(max(gts.max(), preds.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (Test Split)")
        # annotate cells
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="red", fontsize=8
                )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating Confusion Matrix: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    results = experiment_data["nhead_tuning"]["SPR_BENCH"]["results"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    results = {}

# Helper to get best nhead by highest test accuracy
best_nhead = None
best_test_acc = -1
for nhead, data in results.items():
    if data["test_acc"] > best_test_acc:
        best_test_acc = data["test_acc"]
        best_nhead = nhead

# ------------------------------------------------------------------ #
# 1. Accuracy curves
try:
    plt.figure()
    for nhead, data in results.items():
        epochs = np.arange(1, len(data["metrics"]["train_acc"]) + 1)
        plt.plot(
            epochs,
            data["metrics"]["train_acc"],
            marker="o",
            label=f"train nhead={nhead}",
        )
        plt.plot(
            epochs, data["metrics"]["val_acc"], marker="x", label=f"val nhead={nhead}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves (n-head tuning)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2. Loss curves
try:
    plt.figure()
    for nhead, data in results.items():
        epochs = np.arange(1, len(data["losses"]["train_loss"]) + 1)
        plt.plot(
            epochs,
            data["losses"]["train_loss"],
            marker="o",
            label=f"train nhead={nhead}",
        )
        plt.plot(
            epochs, data["losses"]["val_loss"], marker="x", label=f"val nhead={nhead}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves (n-head tuning)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3. Test accuracy bar chart
try:
    plt.figure()
    nheads = list(results.keys())
    test_accs = [results[n]["test_acc"] for n in nheads]
    plt.bar(nheads, test_accs, color="skyblue")
    plt.xlabel("n-head")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH Test Accuracy by n-head")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4. Confusion matrix for best model
try:
    if best_nhead is not None:
        preds = np.array(results[best_nhead]["predictions"])
        gts = np.array(results[best_nhead]["ground_truth"])
        num_classes = len(np.unique(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"SPR_BENCH Confusion Matrix (best nhead={best_nhead})")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

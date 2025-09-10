import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["identity_hidden"]["SPR_BENCH"]
    m = ed["metrics"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = np.arange(1, len(m["train_acc"]) + 1)

# 1. Accuracy curves -------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, m["train_acc"], label="Train Acc")
    plt.plot(epochs, m["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy")
    plt.legend()
    path = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2. Loss curves -----------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
    plt.plot(epochs, m["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3. Rule-Faithfulness Score ----------------------------------------
try:
    plt.figure()
    plt.plot(epochs, m["val_rfs"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("RFS")
    plt.title("SPR_BENCH: Validation Rule-Faithfulness Score")
    path = os.path.join(working_dir, "SPR_BENCH_rfs_curve.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating RFS plot: {e}")
    plt.close()

# 4. Test metrics bar chart -----------------------------------------
try:
    plt.figure()
    metrics = ["Test Acc", "Test RFS"]
    values = [ed["test_acc"], ed["test_rfs"]]
    plt.bar(metrics, values, color=["steelblue", "orange"])
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Final Test Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    path = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# 5. Confusion matrix -----------------------------------------------
try:
    preds = ed["predictions"]
    gts = ed["ground_truth"]
    classes = np.unique(np.concatenate([preds, gts]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["NoSymbolicVector"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    losses_tr = exp["losses"]["train"]
    losses_val = exp["losses"]["val"]
    swa_tr = exp["metrics"]["train_swa"]
    swa_val = exp["metrics"]["val_swa"]
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    epochs = np.arange(1, len(losses_tr) + 1)
    times = np.array(exp["timestamps"]) - exp["timestamps"][0]

    # Plot 1: loss curves
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # Plot 2: SWA curves
    try:
        plt.figure()
        plt.plot(epochs, swa_tr, label="Train")
        plt.plot(epochs, swa_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Training vs Validation SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # Plot 3: confusion matrix
    try:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # Plot 4: histogram of class frequencies
    try:
        plt.figure()
        bins = np.arange(max(gts.max(), preds.max()) + 2) - 0.5
        plt.hist(gts, bins=bins, alpha=0.6, label="Ground Truth")
        plt.hist(preds, bins=bins, alpha=0.6, label="Predictions")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title("SPR_BENCH: Class Frequency Histogram")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_class_hist.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating histogram: {e}")
        plt.close()

    # Plot 5: validation SWA over time
    try:
        plt.figure()
        plt.plot(times, swa_val, marker="o")
        plt.xlabel("Seconds Since Start")
        plt.ylabel("Validation SWA")
        plt.title("SPR_BENCH: Validation SWA vs Time")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_time.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA-time plot: {e}")
        plt.close()

    # ---------- evaluation metric ----------
    test_acc = (preds == gts).mean() if len(gts) else float("nan")
    print(f"Test accuracy (simple): {test_acc:.4f}")

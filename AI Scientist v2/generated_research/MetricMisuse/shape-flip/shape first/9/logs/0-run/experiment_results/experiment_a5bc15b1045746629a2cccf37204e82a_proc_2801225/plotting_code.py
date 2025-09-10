import matplotlib.pyplot as plt
import numpy as np
import os

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["unidirectional_gru"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    losses_train = ed["losses"]["train"]
    losses_val = ed["losses"]["val"]
    val_swa = ed["metrics"]["val"]
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])
    test_swa = ed["metrics"].get("test", None)

    # 1) Training vs Validation Loss
    try:
        plt.figure()
        epochs = range(1, len(losses_train) + 1)
        plt.plot(epochs, losses_train, label="Train Loss")
        plt.plot(epochs, losses_val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) Validation SWA curve
    try:
        plt.figure()
        plt.plot(epochs, val_swa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation SWA per Epoch")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # 3) Prediction vs Ground-Truth distribution (bar chart)
    try:
        plt.figure()
        classes = sorted(set(gts)) if gts else []
        pred_counts = [sum(p == c for p in preds) for c in classes]
        true_counts = [sum(t == c for t in gts) for c in classes]
        x = np.arange(len(classes))
        width = 0.35
        plt.bar(x - width / 2, true_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("SPR_BENCH: Test Set Class Distribution")
        plt.xticks(x, classes)
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_pred_vs_true_counts.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()

    # Print final metric
    if test_swa is not None:
        print(f"Final Test Shape-Weighted Accuracy: {test_swa:.4f}")

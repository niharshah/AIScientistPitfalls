import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def simple_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean() if len(y_true) else float("nan")


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# navigate to the single run present
run_key = "no_position_embedding"
dset_key = "SPR_BENCH"
exp = experiment_data.get(run_key, {}).get(dset_key, {})

# --------------- plot 1: loss curves ---------------
try:
    train_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH – Loss Curves (No Position Embedding)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_no_pos.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# --------------- plot 2: SWA curves ----------------
try:
    train_swa = exp["metrics"]["train_swa"]
    val_swa = exp["metrics"]["val_swa"]
    epochs = np.arange(1, len(train_swa) + 1)
    plt.figure()
    plt.plot(epochs, train_swa, label="Train SWA")
    plt.plot(epochs, val_swa, label="Validation SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH – SWA Curves (No Position Embedding)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_SWA_curves_no_pos.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# --------------- plot 3: confusion matrix ----------
try:
    preds = np.array(exp.get("predictions", []), dtype=int)
    gts = np.array(exp.get("ground_truth", []), dtype=int)
    if preds.size and gts.size:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH – Confusion Matrix (No Position Embedding)")
        plt.xticks(np.arange(num_classes))
        plt.yticks(np.arange(num_classes))
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_no_pos.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Predictions or ground truth unavailable; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# --------------- print summary metrics -------------
try:
    final_val_swa = exp["metrics"]["val_swa"][-1] if exp else float("nan")
    best_val_swa = max(exp["metrics"]["val_swa"]) if exp else float("nan")
    test_acc = simple_accuracy(gts, preds)
    print(f"Final validation SWA: {final_val_swa:.4f}")
    print(f"Best validation SWA : {best_val_swa:.4f}")
    print(f"Test accuracy       : {test_acc:.4f}")
except Exception as e:
    print(f"Error computing summary metrics: {e}")

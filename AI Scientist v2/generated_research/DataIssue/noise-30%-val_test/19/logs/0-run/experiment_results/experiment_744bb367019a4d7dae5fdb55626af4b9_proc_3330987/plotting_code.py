import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    # Extract logged data
    track = experiment_data["num_epochs"]["SPR_BENCH"]
    epochs = track["epochs"]
    train_loss, val_loss = track["losses"]["train"], track["losses"]["val"]
    train_f1, val_f1 = (
        track["metrics"]["train_macro_f1"],
        track["metrics"]["val_macro_f1"],
    )
    preds, gts = np.array(track["predictions"]), np.array(track["ground_truth"])

    # Compute evaluation metric
    try:
        test_macro_f1 = f1_score(gts, preds, average="macro")
        print(f"Test Macro-F1: {test_macro_f1:.4f}")
    except Exception as e:
        print(f"Error computing test F1: {e}")

    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve_replot.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- Plot 2: Macro-F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ---------- Plot 3: Prediction vs Ground Truth counts ----------
    try:
        plt.figure()
        width = 0.35
        labels = ["Class 0", "Class 1"]
        gt_counts = [np.sum(gts == 0), np.sum(gts == 1)]
        pred_counts = [np.sum(preds == 0), np.sum(preds == 1)]
        x = np.arange(len(labels))
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xticks(x, labels)
        plt.ylabel("Count")
        plt.title("SPR_BENCH: Label Distribution—Ground Truth vs Predictions")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_label_distribution.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating label distribution plot: {e}")
        plt.close()

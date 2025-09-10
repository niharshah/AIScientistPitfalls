import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    epochs = np.arange(1, len(ed["metrics"]["train_macroF1"]) + 1)

    # --------- 1) F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_macroF1"], label="Train Macro-F1")
        plt.plot(epochs, ed["metrics"]["val_macroF1"], label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # --------- 2) Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # --------- 3) Additional validation metrics ----------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["val_SWA"], label="Shape-Wtd Acc (SWA)")
        plt.plot(epochs, ed["metrics"]["val_CWA"], label="Color-Wtd Acc (CWA)")
        plt.plot(epochs, ed["metrics"]["val_SC_Gmean"], label="SC-Gmean")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Additional Validation Metrics")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # --------- 4) Confusion matrix ----------
    try:
        preds = np.array(ed["predictions"])
        trues = np.array(ed["ground_truth"])
        n_labels = preds.max() + 1 if preds.size else 0
        if n_labels > 0:
            cm, _, _ = np.histogram2d(
                trues,
                preds,
                bins=(n_labels, n_labels),
                range=[[0, n_labels], [0, n_labels]],
            )
            plt.figure()
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

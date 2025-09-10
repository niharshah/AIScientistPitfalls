import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- Load experiment data & print test metric ---------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data, spr = None, None

if spr:
    print(f"Test Macro-F1 (SPR_BENCH): {spr.get('test_macroF1', 'N/A'):.4f}")

    epochs = spr["epochs"]
    tr_loss = spr["losses"]["train"]
    val_loss = spr["losses"]["val"]
    tr_f1 = spr["metrics"]["train"]
    val_f1 = spr["metrics"]["val"]
    preds = spr.get("predictions", [])
    gts = spr.get("ground_truth", [])

    # ------------------------ Loss curve ---------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------------ Macro-F1 curve -----------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 curve: {e}")
        plt.close()

    # ------------------------ Confusion matrix ---------------------------
    try:
        from sklearn.metrics import confusion_matrix

        if preds and gts:
            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                "SPR_BENCH: Normalized Confusion Matrix\n(Left: Ground Truth, Right: Predictions)"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

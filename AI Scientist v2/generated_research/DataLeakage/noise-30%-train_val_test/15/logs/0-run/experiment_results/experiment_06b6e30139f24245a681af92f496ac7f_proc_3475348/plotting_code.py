import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["Remove_Symbolic_Feature_Pathway"]["SPR_BENCH"]
    epochs = ed.get("epochs", [])
    train_loss = ed.get("losses", {}).get("train", [])
    val_loss = ed.get("losses", {}).get("val", [])
    train_f1 = ed.get("metrics", {}).get("train", [])
    val_f1 = ed.get("metrics", {}).get("val", [])
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])
    test_loss = ed.get("test_loss", None)
    test_macroF1 = ed.get("test_macroF1", None)

    # --------------------------------------------------------
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 2) Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH – Training vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 3) Confusion matrix on test set
    try:
        if preds and gts:
            cm = confusion_matrix(gts, preds, labels=sorted(set(gts)))
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap="Blues", colorbar=False)
            plt.title("SPR_BENCH – Test Confusion Matrix")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # print evaluation summary
    print(f"Test Loss: {test_loss:.4f} | Test Macro-F1: {test_macroF1:.4f}")

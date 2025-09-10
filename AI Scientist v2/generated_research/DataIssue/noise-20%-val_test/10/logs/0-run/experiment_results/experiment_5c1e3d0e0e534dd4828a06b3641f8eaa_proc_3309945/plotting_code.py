import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# -------- setup & load -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Only continue if data loaded correctly
if "SPR_BENCH" in experiment_data:
    rec = experiment_data["SPR_BENCH"]
    epochs = range(1, len(rec["losses"]["train"]) + 1)

    # --------- 1) Loss curve ----------
    try:
        plt.figure()
        plt.plot(epochs, rec["losses"]["train"], label="Train Loss")
        plt.plot(epochs, rec["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------- 2) F1 curve ------------
    try:
        plt.figure()
        plt.plot(epochs, rec["metrics"]["train_f1"], label="Train Macro-F1")
        plt.plot(epochs, rec["metrics"]["val_f1"], label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Training vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # --------- 3) Confusion matrix ----
    try:
        y_true = rec["ground_truth"]
        y_pred = rec["predictions"]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix (Test)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------- 4) Interpretable accuracy bar ----
    try:
        plt.figure()
        interp_acc = rec["metrics"]["Interpretable-Accuracy"]
        plt.bar(["Interpretable Accuracy"], [interp_acc])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Interpretable Accuracy on Test")
        for i, v in enumerate([interp_acc]):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_interpretable_accuracy.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating interpretable accuracy bar: {e}")
        plt.close()

    # --------- print quick metrics ----
    print(f"Final Val Macro-F1: {rec['metrics']['val_f1'][-1]:.4f}")
    print(
        f"Test Interpretable Accuracy: {rec['metrics']['Interpretable-Accuracy']:.4f}"
    )
else:
    print("SPR_BENCH data not found in experiment_data.npy")

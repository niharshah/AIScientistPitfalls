import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    try:
        d = experiment_data["Remove_Gating_Mechanism"]["SPR_BENCH"]
        epochs = np.array(d["epochs"])
        tr_loss = np.array(d["losses"]["train"])
        val_loss = np.array(d["losses"]["val"])
        tr_f1 = np.array(d["metrics"]["train"])
        val_f1 = np.array(d["metrics"]["val"])
        test_preds = np.array(d["predictions"])
        test_gts = np.array(d["ground_truth"])
        test_loss = d.get("test_loss", None)
        test_f1 = d.get("test_macroF1", None)
        print(f"Test loss: {test_loss:.4f}, Test macro-F1: {test_f1:.4f}")
    except Exception as e:
        print(f"Error extracting sub-dictionary: {e}")
        experiment_data = None

# ------------------------------------------------------------------ #
if experiment_data is not None:
    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves - Remove_Gating_Mechanism (SPR_BENCH)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("Macro-F1 Curves - Remove_Gating_Mechanism (SPR_BENCH)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # 3. Confusion matrix
    try:
        cm = confusion_matrix(test_gts, test_preds)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix - Remove_Gating_Mechanism (SPR_BENCH)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

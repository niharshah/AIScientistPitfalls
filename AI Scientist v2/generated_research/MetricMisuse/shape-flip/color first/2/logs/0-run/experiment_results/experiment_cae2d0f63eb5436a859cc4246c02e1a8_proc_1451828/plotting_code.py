import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    spr = experiment_data["single_layer_gnn"]["SPR"]

    # Helper to turn list[(epoch,val)] -> arrays
    def to_xy(lst):
        lst = np.array(lst)
        return lst[:, 0], lst[:, 1]

    epochs_loss_tr, loss_tr = to_xy(spr["losses"]["train"])
    epochs_loss_val, loss_val = to_xy(spr["losses"]["val"])
    epochs_pc_tr, pc_tr = to_xy(spr["metrics"]["train"])
    epochs_pc_val, pc_val = to_xy(spr["metrics"]["val"])
    y_true = np.array(spr["ground_truth"])
    y_pred = np.array(spr["predictions"])

    # Overall metrics
    acc = (y_true == y_pred).mean() if len(y_true) else float("nan")
    pcwa_final = pc_val[-1] if len(pc_val) else float("nan")
    print(f"FINAL ACC={acc:.4f}, FINAL PCWA={pcwa_final:.4f}")

    # -------------------------------------------------- Plot 1: Loss
    try:
        plt.figure()
        plt.plot(epochs_loss_tr, loss_tr, label="Train")
        plt.plot(epochs_loss_val, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Training vs Validation Loss\nDataset: SPR")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------------------------------------------- Plot 2: PCWA
    try:
        plt.figure()
        plt.plot(epochs_pc_tr, pc_tr, label="Train")
        plt.plot(epochs_pc_val, pc_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title("Training vs Validation PCWA\nDataset: SPR")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_PCWA_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating PCWA plot: {e}")
        plt.close()

    # -------------------------------------------------- Plot 3: Confusion Matrix
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", color="black")
        plt.title(
            "Confusion Matrix\nDataset: SPR (Left: Ground Truth, Right: Predictions)"
        )
        fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

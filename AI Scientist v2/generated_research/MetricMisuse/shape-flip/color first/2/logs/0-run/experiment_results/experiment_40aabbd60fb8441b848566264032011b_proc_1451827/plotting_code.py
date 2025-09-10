import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
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

if experiment_data:
    ed = experiment_data["ShapeRemoval"]["SPR"]

    # -------------------------------------------------------------- #
    # helper to convert (epoch, value) tuples -> two numpy arrays
    def split_points(pair_list):
        arr = np.array(pair_list)
        return arr[:, 0], arr[:, 1]

    # collect data
    epochs_loss_tr, loss_tr = split_points(ed["losses"]["train"])
    epochs_loss_val, loss_val = split_points(ed["losses"]["val"])
    epochs_pc_tr, pc_tr = split_points(ed["metrics"]["train"])
    epochs_pc_val, pc_val = split_points(ed["metrics"]["val"])
    y_pred, y_true = np.array(ed["predictions"]), np.array(ed["ground_truth"])
    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    # -------------------------------------------------------------- #
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs_loss_tr, loss_tr, label="Train")
        plt.plot(epochs_loss_val, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("ShapeRemoval-SPR: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "ShapeRemoval_SPR_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # 2) PCWA curves
    try:
        plt.figure()
        plt.plot(epochs_pc_tr, pc_tr, label="Train")
        plt.plot(epochs_pc_val, pc_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title("ShapeRemoval-SPR: Training vs Validation PCWA")
        plt.legend()
        fname = os.path.join(working_dir, "ShapeRemoval_SPR_PCWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating PCWA curve: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # 3) Confusion matrix heat map
    try:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title("ShapeRemoval-SPR: Confusion Matrix")
        fname = os.path.join(working_dir, "ShapeRemoval_SPR_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # print final test metrics for quick reference
    try:
        # metrics were printed during experiment run, recompute quickly here
        acc = (y_true == y_pred).mean()
        print(f"Test Accuracy: {acc:.3f}")
        if len(pc_val) > 0:
            print(f"Final Validation PCWA: {pc_val[-1]:.3f}")
    except Exception as e:
        print(f"Error printing metrics: {e}")

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------#
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_val_f1s = {}

for dname, dct in experiment_data.items():
    losses_tr = np.asarray(dct["losses"]["train"])
    losses_val = np.asarray(dct["losses"]["val"])
    f1_tr = np.asarray(dct["metrics"]["train"])
    f1_val = np.asarray(dct["metrics"]["val"])
    best_val_f1s[dname] = float(np.max(f1_val)) if f1_val.size else 0.0

    # --------------------- Loss curves ----------------------------#
    try:
        plt.figure()
        plt.plot(losses_tr, label="Train")
        plt.plot(losses_val, label="Validation")
        plt.title(f"{dname} Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot ({dname}): {e}")
        plt.close()

    # -------------------- F1 curves -------------------------------#
    try:
        plt.figure()
        plt.plot(f1_tr, label="Train")
        plt.plot(f1_val, label="Validation")
        plt.title(f"{dname} Macro-F1 Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot ({dname}): {e}")
        plt.close()

    # ----------------- Test metrics bar chart ---------------------#
    try:
        preds = np.asarray(dct["predictions"][0]).flatten()
        gts = np.asarray(dct["ground_truth"][0]).flatten()
        test_f1 = f1_score(gts, preds, average="macro")
        test_mcc = matthews_corrcoef(gts, preds)

        plt.figure()
        plt.bar(["Macro-F1", "MCC"], [test_f1, test_mcc], color=["steelblue", "orange"])
        plt.ylim(0, 1)
        plt.title(f"{dname} Test Metrics\nMacro-F1 vs MCC")
        for i, v in enumerate([test_f1, test_mcc]):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, f"{dname.lower()}_test_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"{dname} | Test Macro-F1: {test_f1:.4f} | Test MCC: {test_mcc:.4f}")
    except Exception as e:
        print(f"Error creating metrics bar chart ({dname}): {e}")
        plt.close()

    # ----------------- Confusion matrix ---------------------------#
    try:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, str(val), ha="center", va="center", color="black")
        plt.title(f"{dname} Confusion Matrix\nDataset: {dname}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        fname = os.path.join(working_dir, f"{dname.lower()}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix ({dname}): {e}")
        plt.close()

# ----------- Cross-dataset comparison (if >1 datasets) ------------#
if len(best_val_f1s) > 1:
    try:
        plt.figure()
        names, vals = zip(*best_val_f1s.items())
        plt.bar(names, vals, color="green")
        plt.ylim(0, 1)
        plt.title("Best Validation Macro-F1 Across Datasets")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, "cross_dataset_best_val_f1.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset comparison plot: {e}")
        plt.close()

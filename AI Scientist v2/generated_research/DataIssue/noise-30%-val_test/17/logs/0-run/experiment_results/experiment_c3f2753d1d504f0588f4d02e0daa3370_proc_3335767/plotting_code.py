import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
)

# ------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------- #
all_mcc, all_f1, all_dsets = [], [], []

for dname, dct in experiment_data.items():
    losses_tr = np.array(dct["losses"]["train"])
    losses_val = np.array(dct["losses"]["val"])
    mcc_tr = np.array(dct["metrics"]["train"])
    mcc_val = np.array(dct["metrics"]["val"])
    preds = np.array(dct["predictions"][0]).flatten()
    gts = np.array(dct["ground_truth"][0]).flatten()

    # -------------------- Loss curves ------------------------- #
    try:
        plt.figure()
        plt.plot(losses_tr, label="Train")
        plt.plot(losses_val, label="Validation")
        plt.title(f"{dname} BCE Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # -------------------- MCC curves -------------------------- #
    try:
        plt.figure()
        plt.plot(mcc_tr, label="Train")
        plt.plot(mcc_val, label="Validation")
        plt.title(f"{dname} MCC Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews Correlation")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_mcc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot for {dname}: {e}")
        plt.close()

    # ------------------- Test metrics ------------------------- #
    try:
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
        print(f"Error creating test metric plot for {dname}: {e}")
        plt.close()

    # ---------------- Confusion Matrix ------------------------ #
    try:
        cm = confusion_matrix(gts, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title(f"{dname} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        fname = os.path.join(working_dir, f"{dname.lower()}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # ----- collect for cross-dataset comparison --------------- #
    all_f1.append(test_f1)
    all_mcc.append(test_mcc)
    all_dsets.append(dname)

# ------------- Cross-dataset comparison ---------------------- #
if len(all_dsets) > 1:
    try:
        x = np.arange(len(all_dsets))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, all_f1, width, label="Macro-F1")
        plt.bar(x + width / 2, all_mcc, width, label="MCC")
        plt.xticks(x, all_dsets, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title("Dataset Comparison\nMacro-F1 vs MCC")
        plt.legend()
        fname = os.path.join(working_dir, "all_datasets_comparison.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset comparison plot: {e}")
        plt.close()

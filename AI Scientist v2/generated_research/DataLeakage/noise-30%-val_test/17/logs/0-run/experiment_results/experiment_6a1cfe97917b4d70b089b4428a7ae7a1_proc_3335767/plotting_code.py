import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_val_mcc_all = {}

for dname, dct in experiment_data.items():
    losses_tr = np.array(dct["losses"]["train"])
    losses_val = np.array(dct["losses"]["val"])
    mcc_tr = np.array(dct["metrics"]["train"])
    mcc_val = np.array(dct["metrics"]["val"])

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
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------------- MCC curves ------------------------------#
    try:
        plt.figure()
        plt.plot(mcc_tr, label="Train")
        plt.plot(mcc_val, label="Validation")
        plt.title(f"{dname} MCC Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_mcc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot: {e}")
        plt.close()

    # ----------------- Test metrics bar chart ---------------------#
    try:
        preds = np.array(dct["predictions"][0]).flatten()
        gts = np.array(dct["ground_truth"][0]).flatten()
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
        print(f"Error creating metrics bar chart: {e}")
        plt.close()

    # ------------------ Confusion Matrix --------------------------#
    try:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues", vmin=0)
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.title(f"{dname} Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        fname = os.path.join(working_dir, f"{dname.lower()}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # track best val mcc
    try:
        best_val_mcc_all[dname] = float(np.max(mcc_val))
    except Exception:
        pass

# ------------- Cross-dataset comparison (best val MCC) ------------#
try:
    if best_val_mcc_all:
        plt.figure()
        names = list(best_val_mcc_all.keys())
        scores = [best_val_mcc_all[n] for n in names]
        plt.bar(names, scores, color="green")
        plt.ylim(0, 1)
        plt.title("Best Validation MCC Across Datasets")
        for i, v in enumerate(scores):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, "comparison_best_val_mcc.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating cross-dataset comparison: {e}")
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

# ---------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
# ---------------------------------------------------------------#
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}
# ---------------------------------------------------------------#
mcc_summary = {}
for dname, dct in experiment_data.items():
    losses_tr = np.array(dct["losses"]["train"])
    losses_val = np.array(dct["losses"]["val"])
    mcc_tr = np.array(dct["metrics"]["train"])
    mcc_val = np.array(dct["metrics"]["val"])

    # ---------------- Loss curves -------------------------------#
    try:
        plt.figure()
        plt.plot(losses_tr, label="Train")
        plt.plot(losses_val, label="Validation")
        plt.title(f"{dname} Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname.lower()}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # ---------------- MCC curves --------------------------------#
    try:
        plt.figure()
        plt.plot(mcc_tr, label="Train")
        plt.plot(mcc_val, label="Validation")
        plt.title(f"{dname} MCC Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname.lower()}_mcc_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot for {dname}: {e}")
        plt.close()

    # ---------------- Confusion-matrix ---------------------------#
    try:
        preds = np.array(dct["predictions"][0]).flatten()
        gts = np.array(dct["ground_truth"][0]).flatten()
        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(f"{dname} Confusion Matrix\nLeft: Ground Truth, Right: Pred")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname.lower()}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CM for {dname}: {e}")
        plt.close()

    # ---------------- Metrics print & summary --------------------#
    test_f1 = f1_score(gts, preds, average="macro")
    test_mcc = matthews_corrcoef(gts, preds)
    print(f"{dname} | Test Macro-F1: {test_f1:.4f} | Test MCC: {test_mcc:.4f}")
    mcc_summary[dname] = test_mcc

# ------------- MCC comparison across datasets -------------------#
if len(mcc_summary) > 1:
    try:
        plt.figure()
        names, vals = zip(*mcc_summary.items())
        plt.bar(names, vals, color="teal")
        plt.ylim(0, 1)
        plt.ylabel("MCC")
        plt.title("Dataset Comparison – Test MCC")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.xticks(rotation=45, ha="right")
        plt.savefig(os.path.join(working_dir, "all_datasets_mcc_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC comparison plot: {e}")
        plt.close()

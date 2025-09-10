import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, matthews_corrcoef

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

# ------------------------------------------------------------------#
for dname, dct in experiment_data.items():
    losses_tr = np.array(dct["losses"]["train"])
    losses_val = np.array(dct["losses"]["val"])
    f1_tr = np.array(dct["metrics"]["train"])
    f1_val = np.array(dct["metrics"]["val"])

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
        print(f"Error creating F1 plot: {e}")
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
        print(f"{dname}  |  Test Macro-F1: {test_f1:.4f}  |  Test MCC: {test_mcc:.4f}")
    except Exception as e:
        print(f"Error creating metrics bar chart: {e}")
        plt.close()

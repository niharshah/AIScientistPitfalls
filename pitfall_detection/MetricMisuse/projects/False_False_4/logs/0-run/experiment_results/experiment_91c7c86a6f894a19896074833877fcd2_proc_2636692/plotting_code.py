import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
if ds_name not in experiment_data:
    print(f"Dataset {ds_name} not found in experiment_data, aborting plots.")
else:
    data = experiment_data[ds_name]

    # -------------------- 1) LOSS CURVES ----------------------------------
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Train")
        plt.plot(data["losses"]["dev"], label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.title(f"{ds_name} – Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------------------- 2) ACCURACY CURVES ------------------------------
    try:
        plt.figure()
        plt.plot(data["metrics"]["train_acc"], label="Train")
        plt.plot(data["metrics"]["dev_acc"], label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} – Accuracy Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot: {e}")
        plt.close()

    # -------------------- 3) RGS CURVE ------------------------------------
    try:
        plt.figure()
        plt.plot(data["metrics"]["dev_rgs"], marker="o")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Dev RGS")
        plt.title(f"{ds_name} – Rule Generalisation Score (Dev)")
        plt.savefig(os.path.join(working_dir, f"{ds_name}_RGS_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RGS curve plot: {e}")
        plt.close()

    # -------------------- function to build confusion matrix --------------
    def confusion_matrix(true, pred):
        n = max(max(true), max(pred)) + 1
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(true, pred):
            cm[t, p] += 1
        return cm

    # -------------------- 4) DEV CONFUSION MATRIX -------------------------
    try:
        true_d = data["ground_truth"]["dev"]
        pred_d = data["predictions"]["dev"]
        cm_dev = confusion_matrix(true_d, pred_d)
        plt.figure()
        plt.imshow(cm_dev, cmap="Blues")
        plt.colorbar()
        plt.title(f"{ds_name} – Confusion Matrix (Dev)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix_dev.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating dev confusion matrix: {e}")
        plt.close()

    # -------------------- 5) TEST CONFUSION MATRIX ------------------------
    try:
        true_t = data["ground_truth"]["test"]
        pred_t = data["predictions"]["test"]
        cm_test = confusion_matrix(true_t, pred_t)
        plt.figure()
        plt.imshow(cm_test, cmap="Greens")
        plt.colorbar()
        plt.title(f"{ds_name} – Confusion Matrix (Test)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix_test.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test confusion matrix: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment dict ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------ helper to count class distribution ---------------
def class_hist(lst, n_cls=2):
    h = [0] * n_cls
    for x in lst:
        if 0 <= x < n_cls:
            h[x] += 1
    return h


# -------------------------- plotting ---------------------------------
for model_name, ds_dict in experiment_data.items():
    for ds_name, rec in ds_dict.items():
        # 1) Loss curves ------------------------------------------------
        try:
            train_loss = rec["losses"]["train"]
            val_loss = rec["losses"]["val"]
            if train_loss and val_loss:
                plt.figure()
                plt.plot(train_loss, label="train")
                plt.plot(val_loss, label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Loss Curves – {model_name} | Dataset: {ds_name}")
                plt.legend()
                fname = f"{model_name}_{ds_name}_loss_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {model_name}/{ds_name}: {e}")
            plt.close()

        # 2) Accuracy curves -------------------------------------------
        try:
            tr_acc = [m["acc"] for m in rec["metrics"]["train"]]
            va_acc = [m["acc"] for m in rec["metrics"]["val"]]
            if tr_acc and va_acc:
                plt.figure()
                plt.plot(tr_acc, label="train")
                plt.plot(va_acc, label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"Accuracy Curves – {model_name} | Dataset: {ds_name}")
                plt.legend()
                fname = f"{model_name}_{ds_name}_accuracy_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {model_name}/{ds_name}: {e}")
            plt.close()

        # 3) Shape-weighted accuracy curves ----------------------------
        try:
            tr_swa = [m["swa"] for m in rec["metrics"]["train"]]
            va_swa = [m["swa"] for m in rec["metrics"]["val"]]
            if tr_swa and va_swa:
                plt.figure()
                plt.plot(tr_swa, label="train")
                plt.plot(va_swa, label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Shape-Weighted Acc")
                plt.title(f"SWA Curves – {model_name} | Dataset: {ds_name}")
                plt.legend()
                fname = f"{model_name}_{ds_name}_swa_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating swa plot for {model_name}/{ds_name}: {e}")
            plt.close()

        # 4) Test set prediction vs ground truth distribution ----------
        try:
            preds = rec["predictions"]
            gts = rec["ground_truth"]
            if preds and gts:
                pred_hist = class_hist(preds)
                gt_hist = class_hist(gts)
                x = np.arange(len(pred_hist))
                width = 0.35
                plt.figure()
                plt.bar(x - width / 2, gt_hist, width, label="Ground Truth")
                plt.bar(x + width / 2, pred_hist, width, label="Predictions")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title(f"Test Distribution – {model_name} | Dataset: {ds_name}")
                plt.legend()
                fname = f"{model_name}_{ds_name}_test_distribution.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating distribution plot for {model_name}/{ds_name}: {e}")
            plt.close()

        # ------------------- print final test metrics -----------------
        try:
            test_metrics = rec["metrics"]["test"]
            print(f"{model_name}/{ds_name} TEST metrics: {test_metrics}")
        except Exception as e:
            print(f"Error fetching test metrics for {model_name}/{ds_name}: {e}")

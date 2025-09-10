import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------
# load experiment data ------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to dig into dict safely
def get_nested(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


ablation = "RemoveHist"
dataset = "SPR_BENCH"
ed_path = [ablation, dataset]
ed = get_nested(experiment_data, ed_path, {})

loss_train = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])
metrics_tr = ed.get("metrics", {}).get("train", [])
metrics_vl = ed.get("metrics", {}).get("val", [])
test_met = ed.get("metrics", {}).get("test", {})
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# -------------------------------------------------------------------
# Plot 1: Loss curves -------------------------------------------------
try:
    if loss_train and loss_val:
        plt.figure()
        plt.plot(loss_train, label="Train")
        plt.plot(loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset} Loss Curves")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dataset}_loss_curve.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------------------------------------------------------
# Plot 2: Accuracy curves --------------------------------------------
try:
    acc_tr = [m["acc"] for m in metrics_tr] if metrics_tr else []
    acc_vl = [m["acc"] for m in metrics_vl] if metrics_vl else []
    if acc_tr and acc_vl:
        plt.figure()
        plt.plot(acc_tr, label="Train")
        plt.plot(acc_vl, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset} Accuracy Curves")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dataset}_accuracy_curve.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# -------------------------------------------------------------------
# Plot 3: Shape-Weighted Accuracy curves -----------------------------
try:
    swa_tr = [m["swa"] for m in metrics_tr] if metrics_tr else []
    swa_vl = [m["swa"] for m in metrics_vl] if metrics_vl else []
    if swa_tr and swa_vl:
        plt.figure()
        plt.plot(swa_tr, label="Train")
        plt.plot(swa_vl, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Acc")
        plt.title(f"{dataset} Shape-Weighted Accuracy")
        plt.legend()
        save_path = os.path.join(working_dir, f"{dataset}_swa_curve.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# -------------------------------------------------------------------
# Plot 4: Confusion matrix (test) ------------------------------------
try:
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(f"{dataset} Confusion Matrix")
        plt.colorbar()
        save_path = os.path.join(working_dir, f"{dataset}_confusion_matrix.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------------------------------------------------------
# Print final evaluation metrics -------------------------------------
if test_met:
    print(f"Test Accuracy: {test_met.get('acc'):.4f}")
    print(f"Test Shape-Weighted Accuracy: {test_met.get('swa'):.4f}")

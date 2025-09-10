import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------- load data -------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------- gather summary --------------------------------------
summary = {}
for exp_key, exp_dict in experiment_data.get("batch_size", {}).items():
    test_preds = np.array(exp_dict["predictions"])
    test_gt = np.array(exp_dict["ground_truth"])
    acc = (test_preds == test_gt).mean()
    summary[exp_key] = acc
# Print results
print("Test accuracy summary (SPR_BENCH):")
for k, v in summary.items():
    print(f"  {k}: {v:.4f}")
best_exp = max(summary, key=summary.get) if summary else None

# ---------------------- figure 1: Loss curves -------------------------------
try:
    plt.figure()
    for exp_key, exp_dict in experiment_data.get("batch_size", {}).items():
        plt.plot(exp_dict["losses"]["train_loss"], label=f"{exp_key} train")
        plt.plot(exp_dict["losses"]["val_loss"], label=f"{exp_key} val", ls="--")
    plt.title("SPR_BENCH Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------------- figure 2: Accuracy curves ---------------------------
try:
    plt.figure()
    for exp_key, exp_dict in experiment_data.get("batch_size", {}).items():
        plt.plot(exp_dict["metrics"]["train_acc"], label=f"{exp_key} train")
        plt.plot(exp_dict["metrics"]["val_acc"], label=f"{exp_key} val", ls="--")
    plt.title("SPR_BENCH Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ---------------------- figure 3: Confusion matrix for best model ----------
try:
    if best_exp:
        preds = experiment_data["batch_size"][best_exp]["predictions"]
        gts = experiment_data["batch_size"][best_exp]["ground_truth"]
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(f"SPR_BENCH Confusion Matrix (Best {best_exp})")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"spr_bench_confusion_{best_exp}.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

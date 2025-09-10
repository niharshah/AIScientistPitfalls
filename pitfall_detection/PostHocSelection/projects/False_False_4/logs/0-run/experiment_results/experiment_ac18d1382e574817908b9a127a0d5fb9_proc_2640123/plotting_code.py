import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["hist_free"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    losses_tr = exp["losses"]["train"]
    losses_val = exp["losses"]["val"]
    acc_tr = exp["metrics"]["train"]
    acc_val = exp["metrics"]["val"]
    swa_tr = exp["swa"]["train"]
    swa_val = exp["swa"]["val"]
    label_gt_val = np.array(exp["ground_truth"]["val"])
    label_pr_val = np.array(exp["predictions"]["val"])
    test_metrics = exp.get("test_metrics", {})

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(losses_tr, label="Train")
        plt.plot(losses_val, label="Validation")
        plt.title("SPR_BENCH Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(acc_tr, label="Train")
        plt.plot(acc_val, label="Validation")
        plt.title("SPR_BENCH Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3) Shape-weighted accuracy curves
    try:
        plt.figure()
        plt.plot(swa_tr, label="Train")
        plt.plot(swa_val, label="Validation")
        plt.title("SPR_BENCH Shape-Weighted Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 4) Confusion matrix (validation)
    try:
        num_cls = max(label_gt_val.max(), label_pr_val.max()) + 1
        if num_cls <= 20:  # keep it readable
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(label_gt_val, label_pr_val):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title("SPR_BENCH Confusion Matrix (Validation)")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---- print final test metrics ----
    print("Test metrics:", test_metrics)

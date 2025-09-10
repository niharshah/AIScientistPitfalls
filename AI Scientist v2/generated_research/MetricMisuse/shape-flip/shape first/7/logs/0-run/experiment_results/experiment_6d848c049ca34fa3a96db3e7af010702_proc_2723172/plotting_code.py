import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["joint_token_embedding"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    # ---------- helper unpack ----------
    def unpack(k):
        xs, ys = zip(*exp[k]["train"]), zip(*exp[k]["val"])
        tr_ep, tr_val = list(xs[0]), list(xs[1])  # epochs identical for both train/val
        return tr_ep, list(xs[1]), list(ys[1])  # epochs, train_values, val_values

    # ---------- 1. loss curve ----------
    try:
        ep, tr_loss, val_loss = unpack("losses")
        plt.figure()
        plt.plot(ep, tr_loss, label="Train Loss")
        plt.plot(ep, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2. SWA curve ----------
    try:
        ep, tr_swa, val_swa = unpack("metrics")
        plt.figure()
        plt.plot(ep, tr_swa, label="Train SWA")
        plt.plot(ep, val_swa, label="Val SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Training vs Validation SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ---------- 3. confusion matrix ----------
    try:
        gt = np.array(exp["ground_truth"])
        pr = np.array(exp["predictions"])
        cm = np.zeros((2, 2), int)
        for g, p in zip(gt, pr):
            cm[g, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(
            "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
        )
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- 4. class distribution ----------
    try:
        counts_gt = [np.sum(gt == i) for i in [0, 1]]
        counts_pr = [np.sum(pr == i) for i in [0, 1]]
        ind = np.arange(2)
        width = 0.35
        plt.figure()
        plt.bar(ind - width / 2, counts_gt, width, label="Ground Truth")
        plt.bar(ind + width / 2, counts_pr, width, label="Predictions")
        plt.xticks(ind, ["Class 0", "Class 1"])
        plt.ylabel("Count")
        plt.title(
            "SPR_BENCH: Class Distribution\nLeft: Ground Truth, Right: Predictions"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_class_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()

    # ---------- print final metric ----------
    try:
        swa_test = exp["metrics"]["val"][-1][1]  # last stored val SWA
        print(f"Latest Validation SWA: {swa_test:.4f}")
    except Exception as e:
        print(f"Error printing metric: {e}")

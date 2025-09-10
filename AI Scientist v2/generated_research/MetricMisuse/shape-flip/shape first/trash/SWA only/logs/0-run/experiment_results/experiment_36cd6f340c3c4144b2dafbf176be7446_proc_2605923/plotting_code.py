import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# load experiment data
ED = None
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ED = experiment_data["Randomized-Symbolic-Input"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

if ED is not None:
    # 1) Loss curves -------------------------------------------------
    try:
        plt.figure()
        plt.plot(ED["losses"]["train"], label="Train")
        plt.plot(ED["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves (Train vs Val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # helper to extract metric lists safely
    def metric_list(split, key):
        return [m.get(key, np.nan) for m in ED["metrics"][split]]

    # 2) Accuracy curves --------------------------------------------
    try:
        train_acc = metric_list("train", "acc")
        val_acc = metric_list("val", "acc")
        if train_acc and val_acc:
            plt.figure()
            plt.plot(train_acc, label="Train")
            plt.plot(val_acc, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("SPR_BENCH Accuracy Curves (Train vs Val)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot: {e}")
        plt.close()

    # 3) Shape-weighted accuracy curves ------------------------------
    try:
        train_swa = metric_list("train", "swa")
        val_swa = metric_list("val", "swa")
        if train_swa and val_swa:
            plt.figure()
            plt.plot(train_swa, label="Train")
            plt.plot(val_swa, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title("SPR_BENCH Shape-Weighted Accuracy (Train vs Val)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating SWA curve plot: {e}")
        plt.close()

    # 4) Confusion matrix on test ------------------------------------
    try:
        y_true = np.array(ED["ground_truth"])
        y_pred = np.array(ED["predictions"])
        if y_true.size and y_pred.size:
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR_BENCH Confusion Matrix (Test Set)")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # print final evaluation metrics
    test_metrics = ED["metrics"].get("test", {})
    print(
        "FINAL TEST METRICS:",
        f"Accuracy={test_metrics.get('acc', 'N/A'):.3f}",
        f"SWA={test_metrics.get('swa', 'N/A'):.3f}",
    )

import matplotlib.pyplot as plt
import numpy as np
import os

# -------- working directory ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data -------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    tr_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    tr_acc = [m["acc"] for m in data["metrics"]["train"]]
    val_acc = [m["acc"] for m in data["metrics"]["val"]]
    test_acc = data["metrics"]["test"].get("acc")
    test_swa = data["metrics"]["test"].get("swa")
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])

    # 1) Loss curves
    try:
        if tr_loss and val_loss:
            plt.figure()
            plt.plot(range(1, len(tr_loss) + 1), tr_loss, label="train")
            plt.plot(range(1, len(val_loss) + 1), val_loss, label="val")
            plt.title("SPR_BENCH – Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        if tr_acc and val_acc:
            plt.figure()
            plt.plot(range(1, len(tr_acc) + 1), tr_acc, label="train")
            plt.plot(range(1, len(val_acc) + 1), val_acc, label="val")
            plt.title("SPR_BENCH – Training vs Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # 3) Test metric bar chart
    try:
        if test_acc is not None and test_swa is not None:
            plt.figure()
            plt.bar(
                ["Accuracy", "Shape-Weighted Acc"],
                [test_acc, test_swa],
                color=["skyblue", "salmon"],
            )
            plt.title("SPR_BENCH – Test Metrics")
            plt.ylabel("Score")
            fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating test metric bar: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        if preds and gts:
            cm = np.zeros((2, 2), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(
                        j, i, str(cm[i, j]), ha="center", va="center", color="black"
                    )
            plt.title("SPR_BENCH – Confusion Matrix (Test)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.colorbar()
            fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -------- print summary -----------
    print(f"Test Accuracy: {test_acc:.3f} | Shape-Weighted Accuracy: {test_swa:.3f}")
else:
    print("No SPR_BENCH data found in experiment_data.npy")

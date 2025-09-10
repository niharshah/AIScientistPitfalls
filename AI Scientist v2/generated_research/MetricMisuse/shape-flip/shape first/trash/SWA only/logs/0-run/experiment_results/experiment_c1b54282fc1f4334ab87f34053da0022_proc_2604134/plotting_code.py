import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    ed = experiment_data["SPR_BENCH"]

    # ----------- gather series -------------
    train_acc = [m["acc"] for m in ed["metrics"]["train"]]
    val_acc = [m["acc"] for m in ed["metrics"]["val"]]
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    epochs = range(1, len(train_acc) + 1)

    test_acc = ed["metrics"]["test"].get("acc")
    test_swa = ed["metrics"]["test"].get("swa")
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))

    # 1) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Validation Acc")
        plt.title("SPR_BENCH – Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.title("SPR_BENCH – Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3) Test metrics bar chart
    try:
        plt.figure()
        plt.bar(
            ["Accuracy", "SWA"], [test_acc, test_swa], color=["steelblue", "orange"]
        )
        plt.title("SPR_BENCH – Test Metrics")
        plt.ylabel("Score")
        plt.savefig(os.path.join(working_dir, "spr_bench_test_metrics_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        if preds.size and gts.size:
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(gts, preds):
                cm[int(t), int(p)] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(
                        j, i, str(cm[i, j]), ha="center", va="center", color="black"
                    )
            plt.title(
                "SPR_BENCH – Confusion Matrix\nLeft: Ground Truth rows, Right: Predicted cols"
            )
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.xticks([0, 1])
            plt.yticks([0, 1])
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # -------- print metrics -------------
    print(f"Final TEST: Accuracy={test_acc:.3f}, SWA={test_swa:.3f}")
else:
    print("No SPR_BENCH data found in experiment_data.")

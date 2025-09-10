import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data.get("SPR_BENCH", None)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    # ---------- helpers ----------
    def safe_close():
        if plt.get_fignums():
            plt.close()

    # -------- 1) loss curves -----
    try:
        plt.figure()
        plt.plot(ed["losses"]["train"], label="train")
        plt.plot(ed["losses"]["val"], label="val")
        plt.title("SPR_BENCH – Loss vs Epochs\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        safe_close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        safe_close()

    # -------- 2) accuracy curves ---
    try:
        train_acc = [d["acc"] for d in ed["metrics"]["train"]]
        val_acc = [d["acc"] for d in ed["metrics"]["val"]]
        plt.figure()
        plt.plot(train_acc, label="train")
        plt.plot(val_acc, label="val")
        plt.title("SPR_BENCH – Accuracy vs Epochs\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
        plt.savefig(fname)
        safe_close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        safe_close()

    # -------- 3) test metrics bar ---
    try:
        test_acc = ed["metrics"]["test"]["acc"]
        swa = ed["metrics"]["test"]["swa"]
        plt.figure()
        plt.bar(["acc", "swa"], [test_acc, swa], color=["skyblue", "salmon"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH – Test Metrics\nLeft: Accuracy, Right: Shape-Weighted")
        fname = os.path.join(working_dir, "spr_bench_test_metrics_bar.png")
        plt.savefig(fname)
        safe_close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        safe_close()

    # -------- 4) confusion matrix heatmap ----
    try:
        y_true = np.array(ed["ground_truth"])
        y_pred = np.array(ed["predictions"])
        classes = sorted(set(y_true) | set(y_pred))
        if len(classes) == 2:  # only plot if both classes present
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xticks([0, 1], classes)
            plt.yticks([0, 1], classes)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("SPR_BENCH – Confusion Matrix\nLeft: Class 0, Right: Class 1")
            fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
            plt.savefig(fname)
            safe_close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        safe_close()

    # -------- print metrics -------
    print(f"Test accuracy: {ed['metrics']['test']['acc']:.3f}")
    print(f"Shape-weighted accuracy: {ed['metrics']['test']['swa']:.3f}")
else:
    print("No SPR_BENCH data found in experiment_data.")

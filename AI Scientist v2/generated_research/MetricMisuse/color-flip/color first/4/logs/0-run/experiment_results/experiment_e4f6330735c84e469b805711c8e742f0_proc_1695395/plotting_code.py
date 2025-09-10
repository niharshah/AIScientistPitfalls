import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_key = ("frozen_cluster", "SPR_BENCH")
if (
    experiment_data
    and exp_key[0] in experiment_data
    and exp_key[1] in experiment_data[exp_key[0]]
):
    exp = experiment_data[exp_key[0]][exp_key[1]]
else:
    print("Required experiment entry not found — aborting plots.")
    exp = None

# -------- plotting --------
if exp:
    # --- 1. loss curves ---
    try:
        epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train")
        plt.plot(epochs, exp["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --- 2. validation metrics ---
    try:
        mets = exp["metrics"]["val"]
        if mets:
            epochs = np.arange(1, len(mets) + 1)
            acc = [m["acc"] for m in mets]
            cwa = [m["CWA"] for m in mets]
            swa = [m["SWA"] for m in mets]
            comp = [m["CompWA"] for m in mets]
            plt.figure()
            plt.plot(epochs, acc, label="Accuracy")
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, comp, label="CompWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("SPR_BENCH: Validation Metrics over Epochs")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics curve: {e}")
        plt.close()

    # --- 3. confusion matrix ---
    try:
        gt = np.array(exp["ground_truth"])
        pr = np.array(exp["predictions"])
        if gt.size > 0 and pr.size > 0:
            num_classes = int(max(gt.max(), pr.max())) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gt, pr):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
            )
            plt.xticks(range(num_classes))
            plt.yticks(range(num_classes))
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

# ------ paths & data loading ------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_batch_norm"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    # -------- extract series --------
    tr_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    tr_acc = [d["acc"] for d in exp["metrics"]["train"]]
    val_acc = [d["acc"] for d in exp["metrics"]["val"]]
    val_cwa = [d["CompWA"] for d in exp["metrics"]["val"]]
    epochs = list(range(1, len(tr_loss) + 1))
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    n_classes = len(set(gts)) if len(gts) else 0

    # -------- loss curves --------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves (No BatchNorm)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------- accuracy curves --------
    try:
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves (No BatchNorm)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot: {e}")
        plt.close()

    # -------- CompWA curve (validation) --------
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Comp-Weighted Accuracy")
        plt.title("SPR_BENCH Validation CompWA (No BatchNorm)")
        fname = os.path.join(working_dir, "SPR_BENCH_compwa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot: {e}")
        plt.close()

    # -------- confusion matrix --------
    try:
        if n_classes and preds.size:
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.title("SPR_BENCH Confusion Matrix (Test, No BatchNorm)")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -------- terminal metrics print --------
    tst = exp.get("metrics", {}).get("test", {})
    print(
        f"Test Accuracy: {tst.get('acc', 'N/A'):.3f}, "
        f"Test CompWA: {tst.get('CompWA', 'N/A'):.3f}"
    )

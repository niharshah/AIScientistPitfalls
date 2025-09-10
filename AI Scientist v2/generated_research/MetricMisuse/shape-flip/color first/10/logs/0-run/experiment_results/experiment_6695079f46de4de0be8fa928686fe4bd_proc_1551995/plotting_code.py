import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# setup
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    run = experiment_data["SPR_BENCH"]
    # ------------------------------------------------------------------
    # 1. loss curves
    # ------------------------------------------------------------------
    try:
        train_l = run["losses"]["train"]
        val_l = run["losses"]["val"]
        epochs = np.arange(1, len(train_l) + 1)
        plt.figure()
        plt.plot(epochs, train_l, label="Train Loss")
        plt.plot(epochs, val_l, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = "spr_bench_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 2. validation metric curves
    # ------------------------------------------------------------------
    try:
        cwa = [m["cwa"] for m in run["metrics"]["val"]]
        swa = [m["swa"] for m in run["metrics"]["val"]]
        cpx = [m["cpxwa"] for m in run["metrics"]["val"]]
        epochs = np.arange(1, len(cwa) + 1)
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cpx, label="CpxWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Weighted Accuracies")
        plt.legend()
        fname = "spr_bench_val_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 3. test metric bar chart
    # ------------------------------------------------------------------
    try:
        test_m = run["metrics"]["test"]
        metrics = ["cwa", "swa", "cpxwa"]
        vals = [test_m[m] for m in metrics]
        plt.figure()
        plt.bar(metrics, vals)
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Test Weighted Accuracies")
        fname = "spr_bench_test_metric_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 4. confusion matrix
    # ------------------------------------------------------------------
    try:
        preds = run["predictions"]
        trues = run["ground_truth"]
        labels = sorted(set(trues) | set(preds))
        lbl2i = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for g, p in zip(trues, preds):
            cm[lbl2i[g], lbl2i[p]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.tight_layout()
        fname = "spr_bench_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # print stored test metrics
    # ------------------------------------------------------------------
    try:
        print(
            f"Stored TEST metrics -> "
            f"CWA: {test_m['cwa']:.3f}, "
            f"SWA: {test_m['swa']:.3f}, "
            f"CpxWA: {test_m['cpxwa']:.3f}"
        )
    except Exception as e:
        print(f"Error printing test metrics: {e}")

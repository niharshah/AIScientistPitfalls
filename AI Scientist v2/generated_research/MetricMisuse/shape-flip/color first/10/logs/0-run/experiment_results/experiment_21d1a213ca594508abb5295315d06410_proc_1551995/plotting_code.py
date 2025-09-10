import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# set up working directory and load data
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
    spr = experiment_data["SPR_BENCH"]

    # ------------------- 1. Loss curves ----------------------------
    try:
        epochs = np.arange(1, len(spr["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train Loss")
        plt.plot(epochs, spr["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ------------------- 2. Validation metric curves ---------------
    try:
        cwa = [m["cwa"] for m in spr["metrics"]["val"]]
        swa = [m["swa"] for m in spr["metrics"]["val"]]
        cpx = [m["cpxwa"] for m in spr["metrics"]["val"]]
        epochs = np.arange(1, len(cwa) + 1)
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cpx, label="CpxWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(
            "SPR_BENCH Validation Weighted-Accuracy Curves\nCWA/SWA/Complexity-WA"
        )
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_val_weighted_acc.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation metric curves: {e}")
        plt.close()

    # ------------------- 3. Test metric bar chart ------------------
    try:
        test_m = spr["metrics"]["test"]
        metrics = ["cwa", "swa", "cpxwa"]
        values = [test_m[m] for m in metrics]
        plt.figure()
        plt.bar(metrics, values, color=["tab:blue", "tab:orange", "tab:green"])
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Test Metrics\nCWA vs. SWA vs. CpxWA")
        fname = os.path.join(working_dir, "spr_bench_test_metrics_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # ------------------- 4. Confusion matrix -----------------------
    try:
        preds = spr["predictions"]
        golds = spr["ground_truth"]
        labels = sorted(list(set(golds) | set(preds)))
        lbl2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for g, p in zip(golds, preds):
            cm[lbl2idx[g], lbl2idx[p]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.title(
            "SPR_BENCH Test Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
        )
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
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------- 5. Print test metrics ---------------------
    try:
        print(
            f"Stored TEST metrics -> "
            f"CWA: {test_m['cwa']:.3f}, "
            f"SWA: {test_m['swa']:.3f}, "
            f"CpxWA: {test_m['cpxwa']:.3f}"
        )
    except Exception as e:
        print(f"Error printing test metrics: {e}")

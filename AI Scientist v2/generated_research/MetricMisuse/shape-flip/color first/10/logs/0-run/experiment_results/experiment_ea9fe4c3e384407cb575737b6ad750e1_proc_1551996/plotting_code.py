import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------
# prepare working directory & load data
# -------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]

    # -------- common helpers --------
    epochs = np.arange(1, len(data["losses"]["train"]) + 1)

    # 1. loss curves --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. weighted-accuracy curves ------------------------------------------
    try:
        cwa = [m["cwa"] for m in data["metrics"]["val"]]
        swa = [m["swa"] for m in data["metrics"]["val"]]
        cpx = [m["cpx"] for m in data["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, cwa, marker="o", label="CWA")
        plt.plot(epochs, swa, marker="s", label="SWA")
        plt.plot(epochs, cpx, marker="^", label="CpxWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH: Validation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_weighted_acc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # 3. confusion matrix ---------------------------------------------------
    try:
        preds = data["predictions"]
        golds = data["ground_truth"]
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

    # 4. print stored test metrics -----------------------------------------
    try:
        tm = data["metrics"]["test"]
        print(
            f"TEST METRICS -> CWA: {tm['cwa']:.3f}, SWA: {tm['swa']:.3f}, CpxWA: {tm['cpx']:.3f}"
        )
    except Exception as e:
        print(f"Error printing test metrics: {e}")

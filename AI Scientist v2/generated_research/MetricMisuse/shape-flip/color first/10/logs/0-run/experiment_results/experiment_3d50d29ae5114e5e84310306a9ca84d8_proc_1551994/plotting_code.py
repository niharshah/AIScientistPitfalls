import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# set up working directory
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]

    # -------------------- 1. Loss curves ---------------------------
    try:
        losses = data["losses"]
        epochs = np.arange(1, len(losses["train"]) + 1)
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = "spr_bench_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------------- 2. Validation CWA ------------------------
    try:
        cwa = [m["cwa"] for m in data["metrics"]["val"]]
        epochs = np.arange(1, len(cwa) + 1)
        plt.figure()
        plt.plot(epochs, cwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.title("SPR_BENCH: Validation Color-Weighted Accuracy")
        fname = "spr_bench_val_cwa.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # -------------------- 3. Validation SWA ------------------------
    try:
        swa = [m["swa"] for m in data["metrics"]["val"]]
        epochs = np.arange(1, len(swa) + 1)
        plt.figure()
        plt.plot(epochs, swa, marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH: Validation Shape-Weighted Accuracy")
        fname = "spr_bench_val_swa.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # -------------------- 4. Validation CpxWA ----------------------
    try:
        cpx = [m["cpx"] for m in data["metrics"]["val"]]
        epochs = np.arange(1, len(cpx) + 1)
        plt.figure()
        plt.plot(epochs, cpx, marker="o", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.title("SPR_BENCH: Validation Complexity-Weighted Accuracy")
        fname = "spr_bench_val_cpx.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CpxWA plot: {e}")
        plt.close()

    # -------------------- 5. Confusion Matrix ----------------------
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
        fname = "spr_bench_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -------------------- Print stored test metrics ----------------
    try:
        tm = data["metrics"]["test"]
        print(
            f"Stored TEST metrics -> CWA: {tm['cwa']:.3f}, "
            f"SWA: {tm['swa']:.3f}, "
            f"CpxWA: {tm['cpx']:.3f}"
        )
    except Exception as e:
        print(f"Error printing test metrics: {e}")

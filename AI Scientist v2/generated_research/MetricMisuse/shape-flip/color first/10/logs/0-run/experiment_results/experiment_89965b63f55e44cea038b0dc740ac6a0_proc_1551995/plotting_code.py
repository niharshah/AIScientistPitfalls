import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------
# setup
# --------------------------------------------------------------
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

    # ---------------------------------------------------------- #
    # 1. loss curves                                             #
    # ---------------------------------------------------------- #
    try:
        train_losses = np.array(run["losses"]["train"])
        val_losses = np.array(run["losses"]["val"])
        if train_losses.size and val_losses.size:
            epochs = np.arange(1, len(train_losses) + 1)
            plt.figure()
            plt.plot(epochs, train_losses, label="Train Loss")
            plt.plot(epochs, val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("SPR_BENCH Loss Curves")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------------------------------------------------------- #
    # 2. validation weighted-accuracy curves                     #
    # ---------------------------------------------------------- #
    try:
        val_metrics = run["metrics"]["val"]
        if val_metrics:
            cwa = [m["cwa"] for m in val_metrics]
            swa = [m["swa"] for m in val_metrics]
            cpx = [m["cpxwa"] for m in val_metrics]
            epochs = np.arange(1, len(cwa) + 1)
            # optional thinning if >5 epochs to at most 5 ticks
            tick_idx = np.linspace(0, len(epochs) - 1, min(5, len(epochs)), dtype=int)
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cpx, label="CpxWA")
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title("SPR_BENCH Validation Metrics")
            plt.xticks(epochs[tick_idx])
            plt.legend()
            plt.savefig(os.path.join(working_dir, "spr_bench_val_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # ---------------------------------------------------------- #
    # 3. confusion matrix (test)                                 #
    # ---------------------------------------------------------- #
    try:
        preds = run["predictions"]
        golds = run["ground_truth"]
        if preds and golds:
            labels = sorted(set(golds) | set(preds))
            lbl2i = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for g, p in zip(golds, preds):
                cm[lbl2i[g], lbl2i[p]] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
            plt.title(
                "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
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
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------------------------------------------------------- #
    # 4. print stored test metrics                               #
    # ---------------------------------------------------------- #
    try:
        tm = run["metrics"]["test"]
        print(
            f"TEST METRICS -> CWA: {tm['cwa']:.3f}  SWA: {tm['swa']:.3f}  CpxWA: {tm['cpxwa']:.3f}"
        )
    except Exception as e:
        print(f"Error printing test metrics: {e}")

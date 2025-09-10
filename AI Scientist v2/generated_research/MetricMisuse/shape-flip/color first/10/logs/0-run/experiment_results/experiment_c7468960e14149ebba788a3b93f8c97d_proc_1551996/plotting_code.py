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
    run = experiment_data["SPR_BENCH"]
    # ------------------------------------------------------------------
    # 1. training / validation loss curves
    # ------------------------------------------------------------------
    try:
        train_loss = run.get("losses", {}).get("train", [])
        val_loss = run.get("losses", {}).get("val", [])
        if train_loss and val_loss:
            epochs = np.arange(1, len(train_loss) + 1)
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH: Train vs Validation Loss")
            plt.legend()
            fname = "spr_bench_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 2. validation metric curves (CWA, SWA, CpxWA)
    # ------------------------------------------------------------------
    try:
        val_metrics = run.get("metrics", {}).get("val", [])
        if val_metrics:
            cwa = [m["cwa"] for m in val_metrics]
            swa = [m["swa"] for m in val_metrics]
            cpx = [m["cpxwa"] for m in val_metrics]
            epochs = np.arange(1, len(cwa) + 1)
            plt.figure()
            plt.plot(epochs, cwa, marker="o", label="CWA")
            plt.plot(epochs, swa, marker="s", label="SWA")
            plt.plot(epochs, cpx, marker="^", label="CpxWA")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("SPR_BENCH: Validation Accuracies (CWA/SWA/CpxWA)")
            plt.legend()
            fname = "spr_bench_validation_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 3. confusion matrix on test set
    # ------------------------------------------------------------------
    try:
        preds = run.get("predictions", [])
        gts = run.get("ground_truth", [])
        if preds and gts and len(preds) == len(gts):
            labels = sorted(set(gts) | set(preds))
            lbl2idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[lbl2idx[gt], lbl2idx[pr]] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
            plt.title(
                "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            # annotate cells
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

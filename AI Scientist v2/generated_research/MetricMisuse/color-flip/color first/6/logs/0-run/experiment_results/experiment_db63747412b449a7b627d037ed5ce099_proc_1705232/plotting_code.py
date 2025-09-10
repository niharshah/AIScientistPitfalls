import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    model_key = "shape_only"
    data_key = "SPR"

    edata = experiment_data.get(model_key, {}).get(data_key, {})

    # ---------------------------------------------------------------------
    # Plot 1: Loss curves
    try:
        tr = edata["losses"]["train"]  # list of (epoch, loss)
        va = edata["losses"]["val"]
        epochs_tr, loss_tr = zip(*tr) if tr else ([], [])
        epochs_va, loss_va = zip(*va) if va else ([], [])

        plt.figure()
        plt.plot(epochs_tr, loss_tr, label="Train Loss")
        plt.plot(epochs_va, loss_va, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR Loss Curve — Train vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve_shape_only.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # Plot 2: Validation metrics
    try:
        metrics = edata["metrics"]["val"]  # list of (epoch, dict)
        if metrics:
            ep, mdict = zip(*metrics)
            cwa = [d["CWA"] for d in mdict]
            swa = [d["SWA"] for d in mdict]
            pcwa = [d["PCWA"] for d in mdict]

            plt.figure()
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, pcwa, label="PCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("SPR Validation Metrics Over Epochs")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_metric_curves_shape_only.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ---------------------------------------------------------------------
    # Plot 3: Confusion matrix on test split
    try:
        y_true = edata.get("ground_truth", [])
        y_pred = edata.get("predictions", [])
        if y_true and y_pred:
            labels = sorted(list(set(y_true) | set(y_pred)))
            lab2idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[lab2idx[t], lab2idx[p]] += 1

            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR Confusion Matrix — Test Set")
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
            fname = os.path.join(working_dir, "SPR_confusion_matrix_shape_only.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

            # print simple accuracy
            acc = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0.0
            print(f"Test accuracy: {acc:.4f}")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

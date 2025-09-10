import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested keys
def _get(dic, *ks, default=None):
    for k in ks:
        dic = dic.get(k, {})
    return dic if dic != {} else default


# Iterate over datasets (here only SPR_BENCH)
for dname, ddata in experiment_data.items():
    # ---------- Plot 1: Loss curves ----------
    try:
        train_losses = _get(ddata, "losses", "train", default=[])
        val_losses = _get(ddata, "losses", "val", default=[])
        if train_losses and val_losses:
            epochs = range(1, len(train_losses) + 1)
            plt.figure()
            plt.plot(epochs, train_losses, label="Train")
            plt.plot(epochs, val_losses, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} Loss Curves\nTrain vs Validation")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {dname}: {e}")
        plt.close()

    # ---------- Plot 2: Metric curves ----------
    try:
        val_metrics = _get(ddata, "metrics", "val", default=[])
        if val_metrics:
            epochs = [m["epoch"] for m in val_metrics]
            swa = [m.get("swa", np.nan) for m in val_metrics]
            cwa = [m.get("cwa", np.nan) for m in val_metrics]
            hwa = [m.get("hwa", np.nan) for m in val_metrics]
            plt.figure()
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, hwa, label="HWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dname} Validation Metrics\nSWA / CWA / HWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_metric_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error plotting metric curves for {dname}: {e}")
        plt.close()

    # ---------- Plot 3: Confusion matrix ----------
    try:
        preds = np.array(_get(ddata, "predictions", default=[]))
        truths = np.array(_get(ddata, "ground_truth", default=[]))
        if preds.size and truths.size:
            labels = sorted(set(truths) | set(preds))
            idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(truths, preds):
                cm[idx[t], idx[p]] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dname} Confusion Matrix\nRows: Ground Truth, Columns: Predictions"
            )
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()

    # ---------- Plot 4: Label distribution ----------
    try:
        preds = np.array(_get(ddata, "predictions", default=[]))
        truths = np.array(_get(ddata, "ground_truth", default=[]))
        if preds.size and truths.size:
            labels = sorted(set(truths) | set(preds))
            pred_counts = [np.sum(preds == l) for l in labels]
            truth_counts = [np.sum(truths == l) for l in labels]
            x = np.arange(len(labels))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, truth_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pred_counts, width, label="Predictions")
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Count")
            plt.title(f"{dname} Label Frequency\nGround Truth vs Predictions")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_label_distribution.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error plotting label distribution for {dname}: {e}")
        plt.close()

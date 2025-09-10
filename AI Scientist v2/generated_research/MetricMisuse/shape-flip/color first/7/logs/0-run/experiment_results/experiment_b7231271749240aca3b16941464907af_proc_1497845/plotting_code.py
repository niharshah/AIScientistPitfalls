import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------- helpers ----------
def colour_of(tok):
    return tok[1:] if len(tok) > 1 else ""


def shape_of(tok):
    return tok[0]


def cwa(seqs, y_true, y_pred):
    w = [len(set(colour_of(t) for t in s.split())) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def swa(seqs, y_true, y_pred):
    w = [len(set(shape_of(t) for t in s.split())) for s in seqs]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


def cswa(seqs, y_true, y_pred):
    w = [
        len(set(colour_of(t) for t in s.split()))
        + len(set(shape_of(t) for t in s.split()))
        for s in seqs
    ]
    return sum(wi for wi, t, p in zip(w, y_true, y_pred) if t == p) / max(1, sum(w))


# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset, d in experiment_data.items():
    epochs = range(1, len(d["losses"]["train"]) + 1)

    # --------- Loss curve ---------
    try:
        plt.figure()
        plt.plot(epochs, d["losses"]["train"], label="Train")
        plt.plot(epochs, d["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset} – Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # --------- Accuracy curve ---------
    try:
        tr_acc = [m["acc"] for m in d["metrics"]["train"]]
        val_acc = [m["acc"] for m in d["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dset} – Accuracy Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {dset}: {e}")
        plt.close()

    # --------- Structural metrics curve ---------
    try:
        cwa_vals = [m["CWA"] for m in d["metrics"]["val"]]
        swa_vals = [m["SWA"] for m in d["metrics"]["val"]]
        cswa_vals = [m["CSWA"] for m in d["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, cswa_vals, label="CSWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(f"{dset} – Structural Metrics (Validation)")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_struct_metrics_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating structural metrics curve for {dset}: {e}")
        plt.close()

    # --------- Confusion matrix (test) ---------
    try:
        y_pred = np.array(d["predictions"])
        y_true = np.array(d["ground_truth"])
        if y_pred.size and y_true.size:
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset} – Confusion Matrix (Test)")
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

    # --------- Print final test metrics ---------
    try:
        if y_pred.size and y_true.size:
            acc = (y_pred == y_true).mean()
            cwa_ = cwa(d["sequences"], y_true.tolist(), y_pred.tolist())
            swa_ = swa(d["sequences"], y_true.tolist(), y_pred.tolist())
            cswa_ = cswa(d["sequences"], y_true.tolist(), y_pred.tolist())
            print(
                f"{dset} TEST  Acc:{acc:.3f}  CWA:{cwa_:.3f}  SWA:{swa_:.3f}  CSWA:{cswa_:.3f}"
            )
    except Exception as e:
        print(f"Error computing test metrics for {dset}: {e}")

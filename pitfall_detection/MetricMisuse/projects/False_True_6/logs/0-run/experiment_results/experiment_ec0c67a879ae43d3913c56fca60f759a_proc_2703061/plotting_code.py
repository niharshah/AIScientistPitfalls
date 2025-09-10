import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load experiment artefacts ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_HYBRID" in experiment_data:
    rec = experiment_data["SPR_HYBRID"]

    # ---------- 1) learning loss curves ----------
    try:
        epochs = np.arange(1, len(rec["losses"]["train"]) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rec["losses"]["train"], "b-o", label="Train Loss")
        plt.plot(epochs, rec["losses"]["val"], "r-o", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_HYBRID Loss Curves (Train vs Val)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_HYBRID_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------- 2) accuracy curves ----------
    try:
        val_metrics = rec["metrics"]["val"]
        val_acc = [m.get("acc") for m in val_metrics]
        val_swa = [m.get("swa") for m in val_metrics]
        epochs = np.arange(1, len(val_acc) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_acc, "g-s", label="Val Accuracy")
        plt.plot(epochs, val_swa, "m-^", label="Val Shape-Weighted Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title("SPR_HYBRID Validation Accuracy Metrics")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_HYBRID_accuracy_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

    # ---------- 3) confusion matrix (test) ----------
    try:
        y_true = np.array(rec["ground_truth"])
        y_pred = np.array(rec["predictions"])
        if y_true.size and y_pred.size and y_true.shape == y_pred.shape:
            labels = sorted(set(y_true))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plt.figure(figsize=(5, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(labels)
            plt.yticks(labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_HYBRID Test Confusion Matrix")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_HYBRID_confusion_matrix.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        else:
            print("Confusion matrix skipped: prediction or ground truth missing.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

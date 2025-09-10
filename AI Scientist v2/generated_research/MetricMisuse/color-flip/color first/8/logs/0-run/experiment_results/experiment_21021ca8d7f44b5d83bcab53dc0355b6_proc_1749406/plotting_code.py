import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = data["NoKMeansRawGlyphIDs"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    # --------- reshape helpers -------------------------------------
    def to_xy(lst):
        arr = np.array(lst)
        return arr[:, 0], arr[:, 1:].astype(float) if arr.ndim > 1 else arr[:, 1]

    # --------- 1. Loss curves --------------------------------------
    try:
        epochs_tr, losses_tr = to_xy(exp["losses"]["train"])
        epochs_val, losses_val = to_xy(exp["losses"]["val"])
        plt.figure()
        plt.plot(epochs_tr, losses_tr, label="Train")
        plt.plot(epochs_val, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------- 2. Validation metrics curves ------------------------
    try:
        epochs_m, metrics = to_xy(exp["metrics"]["val"])
        cwa, swa, hm, ocga = metrics.T
        plt.figure()
        plt.plot(epochs_m, cwa, label="CWA")
        plt.plot(epochs_m, swa, label="SWA")
        plt.plot(epochs_m, hm, label="HM")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR: Validation Metrics (CWA/SWA/HM)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_val_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # --------- 3. OCGA curve ---------------------------------------
    try:
        plt.figure()
        plt.plot(epochs_m, ocga, label="OCGA", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("OCGA")
        plt.title("SPR: Validation OCGA over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_val_OCGA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating OCGA plot: {e}")
        plt.close()

    # --------- 4. Confusion matrix ---------------------------------
    try:
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        num_cls = max(preds.max(), gts.max()) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        plt.figure()
        im = plt.imshow(cm_norm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"SPR: Confusion Matrix (Accuracy={np.mean(preds==gts):.3f})")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8,
                )
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

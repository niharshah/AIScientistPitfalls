import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    spr = experiment_data["SPR_BENCH"]

    # ---------- collect per-epoch series ----------
    epochs = range(1, len(spr["losses"]["train"]) + 1)
    tr_loss, val_loss = spr["losses"]["train"], spr["losses"]["val"]
    tr_acc = [m["acc"] for m in spr["metrics"]["train"]]
    val_acc = [m["acc"] for m in spr["metrics"]["val"]]
    tr_swa = [m["swa"] for m in spr["metrics"]["train"]]
    val_swa = [m["swa"] for m in spr["metrics"]["val"]]

    # ------------------- figure 1: loss curves ------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title("SPR_BENCH – Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------- figure 2: accuracy curves --------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.title("SPR_BENCH – Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot: {e}")
        plt.close()

    # --------------- figure 3: shape-weighted accuracy ---------------
    try:
        plt.figure()
        plt.plot(epochs, tr_swa, label="Train")
        plt.plot(epochs, val_swa, label="Validation")
        plt.title("SPR_BENCH – Shape-Weighted Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve plot: {e}")
        plt.close()

    # ------------------- figure 4: confusion matrix -------------------
    try:
        y_true = np.array(spr["ground_truth"])
        y_pred = np.array(spr["predictions"])
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title("SPR_BENCH – Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ------------------------ print test metrics ----------------------
    tst = spr["metrics"]["test"]
    print(
        f"TEST: Accuracy={tst.get('acc', 'N/A'):.3f}, SWA={tst.get('swa', 'N/A'):.3f}"
    )

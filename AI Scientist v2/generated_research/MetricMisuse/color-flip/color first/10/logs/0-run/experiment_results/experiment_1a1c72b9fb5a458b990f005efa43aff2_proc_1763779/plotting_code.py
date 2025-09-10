import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data -----------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data["Orderless"]["SPR_BENCH"]
    m = data["metrics"]
    epochs = np.arange(1, len(m["train_loss"]) + 1)

    # ------------------- 1. loss curves -----------------------------
    try:
        plt.figure()
        plt.plot(epochs, m["train_loss"], label="Train Loss")
        plt.plot(epochs, m["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "orderless_spr_bench_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------- 2. val metric curves -----------------------
    try:
        plt.figure()
        plt.plot(epochs, m["val_CWA"], label="CWA")
        plt.plot(epochs, m["val_SWA"], label="SWA")
        plt.plot(epochs, m["val_CWA2"], label="CWA2")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Validation Weighted Accuracies")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "orderless_spr_bench_val_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # ------------------- 3. confusion matrix ------------------------
    try:
        preds = np.array(data["predictions"])
        gts = np.array(data["ground_truth"])
        tp = np.sum((preds == 1) & (gts == 1))
        tn = np.sum((preds == 0) & (gts == 0))
        fp = np.sum((preds == 1) & (gts == 0))
        fn = np.sum((preds == 0) & (gts == 1))
        cm = np.array([[tn, fp], [fn, tp]])

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(
            "SPR_BENCH – Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
        )
        plt.tight_layout()
        fname = os.path.join(working_dir, "orderless_spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ------------------- print final metrics ------------------------
    last = -1  # convenience alias
    print(f"Final Val Loss : {m['val_loss'][last]:.4f}")
    print(f"Final CWA      : {m['val_CWA'][last]:.4f}")
    print(f"Final SWA      : {m['val_SWA'][last]:.4f}")
    print(f"Final CWA2     : {m['val_CWA2'][last]:.4f}")

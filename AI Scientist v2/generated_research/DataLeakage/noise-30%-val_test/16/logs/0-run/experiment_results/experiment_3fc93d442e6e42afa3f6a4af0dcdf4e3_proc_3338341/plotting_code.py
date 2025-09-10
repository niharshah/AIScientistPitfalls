import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

spr = experiment_data.get("SPR_BENCH", {})
epochs = spr.get("epochs", [])


# -------- helper --------
def confusion_counts(y_true, y_pred):
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    return np.array([[tn, fp], [fn, tp]])


# -------- 1) Loss curves --------
try:
    plt.figure()
    plt.plot(epochs, spr["losses"]["train"], label="Train")
    plt.plot(epochs, spr["losses"]["val"], label="Validation")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    fname = "spr_bench_loss.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------- 2) MCC curves --------
try:
    plt.figure()
    plt.plot(epochs, spr["metrics"]["train_MCC"], label="Train")
    plt.plot(epochs, spr["metrics"]["val_MCC"], label="Validation")
    plt.title("SPR_BENCH MCC Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend()
    fname = "spr_bench_mcc.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating MCC plot: {e}")
    plt.close()

# -------- 3) Rule-Macro-Accuracy curves --------
try:
    plt.figure()
    plt.plot(epochs, spr["metrics"]["train_RMA"], label="Train")
    plt.plot(epochs, spr["metrics"]["val_RMA"], label="Validation")
    plt.title("SPR_BENCH Rule-Macro-Accuracy\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("RMA")
    plt.legend()
    fname = "spr_bench_rma.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating RMA plot: {e}")
    plt.close()

# -------- 4) Confusion matrix --------
try:
    y_true = spr["ground_truth"]
    y_pred = spr["predictions"]
    cm = confusion_counts(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
    plt.xticks([0, 1], ["Neg", "Pos"])
    plt.yticks([0, 1], ["Neg", "Pos"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    fname = "spr_bench_confusion_matrix.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------- 5) Prediction distribution --------
try:
    preds = np.array(y_pred)
    trues = np.array(y_true)
    plt.figure()
    plt.hist(preds[trues == 0], bins=2, alpha=0.7, label="True Negatives")
    plt.hist(preds[trues == 1], bins=2, alpha=0.7, label="True Positives")
    plt.title("SPR_BENCH Prediction Distribution\nLeft: True Neg, Right: True Pos")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.legend()
    fname = "spr_bench_pred_distribution.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating prediction distribution plot: {e}")
    plt.close()

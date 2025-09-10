import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

epochs = spr.get("epochs", list(range(len(spr["losses"]["train"]))))


# ---------- helper ----------
def confusion_counts(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


# ---------- 1) BCE loss ----------
try:
    plt.figure()
    plt.plot(epochs, spr["losses"]["train"], label="Train")
    plt.plot(epochs, spr["losses"]["val"], label="Validation")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    fname = "spr_bench_loss_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2) MCC ----------
try:
    plt.figure()
    plt.plot(epochs, spr["metrics"]["train"], label="Train")
    plt.plot(epochs, spr["metrics"]["val"], label="Validation")
    plt.title("SPR_BENCH MCC Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews Corr Coef")
    plt.legend()
    fname = "spr_bench_mcc_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating MCC plot: {e}")
    plt.close()

# ---------- 3) RMA ----------
try:
    plt.figure()
    plt.plot(epochs, spr["RMA"]["train"], label="Train")
    plt.plot(epochs, spr["RMA"]["val"], label="Validation")
    plt.title("SPR_BENCH Rule-Macro Accuracy\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("RMA")
    plt.legend()
    fname = "spr_bench_rma_curves.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating RMA plot: {e}")
    plt.close()

# ---------- 4) Confusion matrix ----------
try:
    y_true = np.array(spr["ground_truth"])
    y_pred = np.array(spr["predictions"])
    cm = confusion_counts(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(
        "SPR_BENCH Confusion Matrix on Test\nLeft: Ground Truth, Right: Predictions"
    )
    plt.xticks([0, 1], ["Neg", "Pos"])
    plt.yticks([0, 1], ["Neg", "Pos"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fname = "spr_bench_confusion_matrix.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- 5) Prediction histogram ----------
try:
    plt.figure()
    plt.hist(y_pred[y_true == 0], bins=2, alpha=0.7, label="True Negatives")
    plt.hist(y_pred[y_true == 1], bins=2, alpha=0.7, label="True Positives")
    plt.title("SPR_BENCH Prediction Distribution\nLeft: True Neg, Right: True Pos")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.legend()
    fname = "spr_bench_prediction_histogram.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating prediction histogram: {e}")
    plt.close()

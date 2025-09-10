import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

if "SPR_BENCH" not in experiment_data:
    print("SPR_BENCH entry not found in experiment_data.npy")
    raise SystemExit

d = experiment_data["SPR_BENCH"]
epochs = np.array(d.get("epochs", []))

loss_tr = np.array(d["losses"].get("train", []))
loss_val = np.array(d["losses"].get("val", []))


def _metric_arr(lst, key):
    return np.array([m.get(key, np.nan) for m in lst])


acc_tr = _metric_arr(d["metrics"].get("train", []), "acc")
acc_val = _metric_arr(d["metrics"].get("val", []), "acc")
mcc_tr = _metric_arr(d["metrics"].get("train", []), "MCC")
mcc_val = _metric_arr(d["metrics"].get("val", []), "MCC")

y_true = d.get("ground_truth", [])
y_pred = d.get("predictions", [])


# ---------- helper ----------
def confusion_counts(y_t, y_p):
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_t, y_p))
    tn = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_t, y_p))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_t, y_p))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_t, y_p))
    return np.array([[tn, fp], [fn, tp]])


# ---------- 1) Loss curves ----------
try:
    if len(epochs) and len(loss_tr) and len(loss_val):
        plt.figure()
        plt.plot(epochs, loss_tr, label="Train")
        plt.plot(epochs, loss_val, label="Validation")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = "spr_bench_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2) Accuracy curves ----------
try:
    if len(epochs) and len(acc_tr) and len(acc_val):
        plt.figure()
        plt.plot(epochs, acc_tr, label="Train")
        plt.plot(epochs, acc_val, label="Validation")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = "spr_bench_accuracy_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- 3) MCC curves ----------
try:
    if len(epochs) and len(mcc_tr) and len(mcc_val):
        plt.figure()
        plt.plot(epochs, mcc_tr, label="Train")
        plt.plot(epochs, mcc_val, label="Validation")
        plt.title("SPR_BENCH MCC Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews CorrCoef")
        plt.legend()
        fname = "spr_bench_mcc_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating MCC plot: {e}")
    plt.close()

# ---------- 4) Confusion matrix ----------
try:
    if len(y_true) and len(y_pred):
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
                plt.text(j, i, cm[i, j], ha="center", va="center")
        fname = "spr_bench_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- 5) Prediction histogram ----------
try:
    if len(y_true) and len(y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        plt.figure()
        plt.hist(
            y_pred[y_true == 0],
            bins=np.arange(-0.5, 2),
            alpha=0.7,
            label="True Negatives",
        )
        plt.hist(
            y_pred[y_true == 1],
            bins=np.arange(-0.5, 2),
            alpha=0.7,
            label="True Positives",
        )
        plt.title("SPR_BENCH Prediction Distribution\nLeft: True Neg, Right: True Pos")
        plt.xlabel("Predicted Class")
        plt.ylabel("Count")
        plt.legend()
        fname = "spr_bench_pred_histogram.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating histogram plot: {e}")
    plt.close()

# ---------- print final metrics ----------
if "test_metrics" in d:
    print("\n===== TEST METRICS =====")
    for k, v in d["test_metrics"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

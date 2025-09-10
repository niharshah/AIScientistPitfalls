import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data ------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit


# ------------------- helper ------------------- #
def confusion_counts(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


# ------------------- plotting loop ------------------- #
for ds_name, rec in experiment_data.items():
    epochs = rec.get("epochs", [])
    losses_tr = rec["losses"]["train"]
    losses_val = rec["losses"]["val"]
    acc_tr = [m["acc"] for m in rec["metrics"]["train"]]
    acc_val = [m["acc"] for m in rec["metrics"]["val"]]
    mcc_tr = [m["MCC"] for m in rec["metrics"]["train"]]
    mcc_val = [m["MCC"] for m in rec["metrics"]["val"]]
    y_pred = np.array(rec.get("predictions", []))
    y_true = np.array(rec.get("ground_truth", []))

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = f"{ds_name.lower()}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, acc_tr, label="Train")
        plt.plot(epochs, acc_val, label="Validation")
        plt.title(f"{ds_name} Accuracy Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"{ds_name.lower()}_accuracy_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating acc plot for {ds_name}: {e}")
        plt.close()

    # 3) MCC curves
    try:
        plt.figure()
        plt.plot(epochs, mcc_tr, label="Train")
        plt.plot(epochs, mcc_val, label="Validation")
        plt.title(f"{ds_name} MCC Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        fname = f"{ds_name.lower()}_mcc_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot for {ds_name}: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        if y_true.size and y_pred.size:
            cm = confusion_counts(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title(
                f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.xticks([0, 1], ["Neg", "Pos"])
            plt.yticks([0, 1], ["Neg", "Pos"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = f"{ds_name.lower()}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating CM plot for {ds_name}: {e}")
        plt.close()

    # 5) Prediction histogram
    try:
        if y_true.size and y_pred.size:
            plt.figure()
            plt.hist(y_pred[y_true == 0], bins=2, alpha=0.7, label="True Negatives")
            plt.hist(y_pred[y_true == 1], bins=2, alpha=0.7, label="True Positives")
            plt.title(
                f"{ds_name} Prediction Distribution\nLeft: True Neg, Right: True Pos"
            )
            plt.xlabel("Predicted Class")
            plt.ylabel("Count")
            plt.legend()
            fname = f"{ds_name.lower()}_pred_hist.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating hist plot for {ds_name}: {e}")
        plt.close()

    # Print final test metrics
    test_metrics = rec.get("test_metrics", {})
    if test_metrics:
        print(
            f"{ds_name} test metrics:",
            {k: round(v, 4) for k, v in test_metrics.items()},
        )

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit


def confusion_counts(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


# ---------- iterate over datasets ----------
for dset_name, rec in experiment_data.items():
    epochs = np.array(rec["epochs"])
    tr_loss = np.array(rec["losses"]["train"])
    val_loss = np.array(rec["losses"]["val"])
    tr_acc = np.array([m["acc"] for m in rec["metrics"]["train"]])
    val_acc = np.array([m["acc"] for m in rec["metrics"]["val"]])
    tr_mcc = np.array([m["MCC"] for m in rec["metrics"]["train"]])
    val_mcc = np.array([m["MCC"] for m in rec["metrics"]["val"]])
    y_pred = np.array(rec.get("predictions", []))
    y_true = np.array(rec.get("ground_truth", []))

    # optional epoch thinning for readability
    if len(epochs) > 10:
        idx = np.linspace(0, len(epochs) - 1, 10, dtype=int)
    else:
        idx = slice(None)

    # 1) Loss
    try:
        plt.figure()
        plt.plot(epochs[idx], tr_loss[idx], label="Train")
        plt.plot(epochs[idx], val_loss[idx], label="Validation")
        plt.title(f"{dset_name} Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = f"{dset_name.lower()}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset_name}: {e}")
        plt.close()

    # 2) Accuracy
    try:
        plt.figure()
        plt.plot(epochs[idx], tr_acc[idx], label="Train")
        plt.plot(epochs[idx], val_acc[idx], label="Validation")
        plt.title(f"{dset_name} Accuracy Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"{dset_name.lower()}_accuracy_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting accuracy for {dset_name}: {e}")
        plt.close()

    # 3) MCC
    try:
        plt.figure()
        plt.plot(epochs[idx], tr_mcc[idx], label="Train")
        plt.plot(epochs[idx], val_mcc[idx], label="Validation")
        plt.title(f"{dset_name} MCC Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        fname = f"{dset_name.lower()}_mcc_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting MCC for {dset_name}: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        if y_true.size and y_pred.size:
            cm = confusion_counts(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title(f"{dset_name} Confusion Matrix\nLeft: Ground Truth, Right: Preds")
            plt.xticks([0, 1], ["Neg", "Pos"])
            plt.yticks([0, 1], ["Neg", "Pos"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, int(cm[i, j]), ha="center", va="center")
            fname = f"{dset_name.lower()}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion for {dset_name}: {e}")
        plt.close()

    # 5) Prediction histogram
    try:
        if y_true.size and y_pred.size:
            plt.figure()
            plt.hist(y_pred[y_true == 0], bins=2, alpha=0.7, label="True Neg")
            plt.hist(y_pred[y_true == 1], bins=2, alpha=0.7, label="True Pos")
            plt.title(
                f"{dset_name} Prediction Distribution\nLeft: True Neg, Right: True Pos"
            )
            plt.xlabel("Predicted Class")
            plt.ylabel("Count")
            plt.legend()
            fname = f"{dset_name.lower()}_prediction_hist.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error plotting histogram for {dset_name}: {e}")
        plt.close()

    # ----- print summary metrics -----
    if "test_metrics" in rec:
        tm = rec["test_metrics"]
        print(
            f"{dset_name} test -> loss:{tm['loss']:.4f}  acc:{tm['acc']:.3f}  MCC:{tm['MCC']:.3f}  RMA:{tm['RMA']:.3f}"
        )

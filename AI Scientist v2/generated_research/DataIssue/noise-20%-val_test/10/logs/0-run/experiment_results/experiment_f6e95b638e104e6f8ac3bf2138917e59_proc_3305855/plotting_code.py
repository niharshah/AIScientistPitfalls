import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------- helper ----------
def macro_f1(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lb in labels:
        tp = np.sum((y_true == lb) & (y_pred == lb))
        fp = np.sum((y_true != lb) & (y_pred == lb))
        fn = np.sum((y_true == lb) & (y_pred != lb))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return np.mean(f1s)


# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- plotting ----------
for dname, dct in experiment_data.items():
    losses = dct["losses"]
    metrics = dct["metrics"]
    preds = np.array(dct.get("preds_test", []))
    gts = np.array(dct.get("gts_test", []))

    if preds.size and gts.size:
        print(f"{dname} Test Macro-F1:", macro_f1(gts, preds))

    # 1) Train/Val loss
    try:
        epochs = np.arange(1, len(losses["train"]) + 1)
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train")
        plt.plot(epochs, losses["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname}: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 2) Train/Val F1
    try:
        epochs = np.arange(1, len(metrics["train_f1"]) + 1)
        plt.figure()
        plt.plot(epochs, metrics["train_f1"], label="Train Macro-F1")
        plt.plot(epochs, metrics["val_f1"], label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dname}: Train vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {dname}: {e}")
        plt.close()

    # 3) Confusion matrix
    try:
        if preds.size and gts.size:
            labels = np.unique(np.concatenate([gts, preds]))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dname}: Confusion Matrix (Test Set)")
            plt.xticks(labels)
            plt.yticks(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
            fname = os.path.join(working_dir, f"{dname.lower()}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # 4) Rule extraction accuracy bar
    try:
        if metrics.get("REA_dev") is not None and metrics.get("REA_test") is not None:
            plt.figure()
            plt.bar(["Dev", "Test"], [metrics["REA_dev"], metrics["REA_test"]])
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title(f"{dname}: Rule Extraction Accuracy")
            fname = os.path.join(working_dir, f"{dname.lower()}_rule_accuracy.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating rule accuracy plot for {dname}: {e}")
        plt.close()

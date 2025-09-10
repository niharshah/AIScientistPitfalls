import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helpers ----------
def macro_f1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lb in labels:
        tp = np.sum((y_true == lb) & (y_pred == lb))
        fp = np.sum((y_true != lb) & (y_pred == lb))
        fn = np.sum((y_true == lb) & (y_pred != lb))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


# ---------- plotting ----------
plot_cap = 5  # maximum total figures
plotted = 0

for dname, dct in experiment_data.items():
    if plotted >= plot_cap:
        break

    # ---- 1) loss curves ----
    try:
        if plotted < plot_cap:
            tr_loss = dct["losses"]["train"]
            val_loss = dct["losses"]["val"]
            epochs = np.arange(1, len(tr_loss) + 1)
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname}: Loss Curves\nLeft: Train, Right: Validation")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
            plotted += 1
    except Exception as e:
        print(f"Error plotting loss for {dname}: {e}")
        plt.close()

    # ---- 2) F1 curves ----
    try:
        if plotted < plot_cap:
            tr_f1 = dct["metrics"]["train_f1"]
            val_f1 = dct["metrics"]["val_f1"]
            epochs = np.arange(1, len(tr_f1) + 1)
            plt.figure()
            plt.plot(epochs, tr_f1, label="Train F1")
            plt.plot(epochs, val_f1, label="Validation F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dname}: Macro-F1 Curves\nLeft: Train, Right: Validation")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_f1_curves.png")
            plt.savefig(fname)
            plt.close()
            plotted += 1
    except Exception as e:
        print(f"Error plotting F1 for {dname}: {e}")
        plt.close()

    # ---- 3) confusion matrix ----
    try:
        if plotted < plot_cap and dct.get("gts_test") and dct.get("preds_test"):
            gts = np.array(dct["gts_test"])
            preds = np.array(dct["preds_test"])
            print(f"{dname} Test Macro-F1:", macro_f1(gts, preds))
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
            plotted += 1
    except Exception as e:
        print(f"Error plotting CM for {dname}: {e}")
        plt.close()

    # ---- 4) rule accuracy bar ----
    try:
        if (
            plotted < plot_cap
            and isinstance(dct["metrics"].get("REA_dev"), (int, float))
            and isinstance(dct["metrics"].get("REA_test"), (int, float))
        ):
            rea_dev = dct["metrics"]["REA_dev"]
            rea_test = dct["metrics"]["REA_test"]
            plt.figure()
            plt.bar(["Dev", "Test"], [rea_dev, rea_test], color=["skyblue", "salmon"])
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title(f"{dname}: Rule Extraction Accuracy\nLeft: Dev, Right: Test")
            fname = os.path.join(working_dir, f"{dname.lower()}_rule_accuracy.png")
            plt.savefig(fname)
            plt.close()
            plotted += 1
    except Exception as e:
        print(f"Error plotting REA for {dname}: {e}")
        plt.close()

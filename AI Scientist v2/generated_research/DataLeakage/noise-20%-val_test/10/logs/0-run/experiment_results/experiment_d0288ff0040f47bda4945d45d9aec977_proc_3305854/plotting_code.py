import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
exp_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def macro_f1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
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


# ---------- plotting ----------
for ds_name, ds in experiment_data.items():
    losses = ds.get("losses", {})
    metrics = ds.get("metrics", {})
    preds = np.array(ds.get("predictions", []))
    gts = np.array(ds.get("ground_truth", []))

    # 1) loss curves
    try:
        tr_loss, val_loss = losses.get("train", []), losses.get("val", [])
        if tr_loss and val_loss:
            epochs = np.arange(1, len(tr_loss) + 1)
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name}: Loss Curves")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name.lower()}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) F1 curves
    try:
        tr_f1, val_f1 = metrics.get("train_f1", []), metrics.get("val_f1", [])
        if tr_f1 and val_f1:
            epochs = np.arange(1, len(tr_f1) + 1)
            plt.figure()
            plt.plot(epochs, tr_f1, label="Train")
            plt.plot(epochs, val_f1, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{ds_name}: Macro-F1 Curves")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name.lower()}_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {ds_name}: {e}")
        plt.close()

    # 3) confusion matrix
    try:
        if preds.size and gts.size:
            cm_labels = np.unique(np.concatenate([gts, preds]))
            cm = np.zeros((len(cm_labels), len(cm_labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{ds_name}: Confusion Matrix (Test)")
            plt.xticks(cm_labels)
            plt.yticks(cm_labels)
            for i in range(len(cm_labels)):
                for j in range(len(cm_labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
            plt.savefig(
                os.path.join(working_dir, f"{ds_name.lower()}_confusion_matrix.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # 4) REA bar chart
    try:
        rea_dev = metrics.get("REA_dev")
        rea_test = metrics.get("REA_test")
        if rea_dev is not None and rea_test is not None:
            plt.figure()
            plt.bar(
                ["REA_dev", "REA_test"], [rea_dev, rea_test], color=["orange", "green"]
            )
            plt.ylim(0, 1)
            plt.title(f"{ds_name}: Rule Extraction Accuracy")
            plt.savefig(os.path.join(working_dir, f"{ds_name.lower()}_rea_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating REA plot for {ds_name}: {e}")
        plt.close()

    # ---------- print overall test macro-F1 ----------
    if preds.size and gts.size:
        print(f"{ds_name} Test Macro-F1:", macro_f1(gts, preds))

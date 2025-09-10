import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ---------------- PATH & DATA -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# --------- containers for aggregate plots -----
final_accs, irfs, dnames = [], [], []

# ------------------ PER-DATASET PLOTS ---------
for dname, dd in experiment_data.items():
    losses = dd.get("losses", {})
    metrics = dd.get("metrics", {})
    y_pred = np.asarray(dd.get("predictions", []))
    y_true = np.asarray(dd.get("ground_truth", []))

    # 1) Loss curves (train & val)
    try:
        tr, val = losses.get("train", []), losses.get("val", [])
        if len(val):
            plt.figure()
            if len(tr):
                plt.plot(range(1, len(tr) + 1), tr, label="Train")
            plt.plot(range(1, len(val) + 1), val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Loss (train vs val)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error loss curve {dname}: {e}")
        plt.close()

    # 2) Accuracy curves (train & val)
    try:
        tr, val = metrics.get("train", []), metrics.get("val", [])
        if len(val):
            plt.figure()
            if len(tr):
                plt.plot(range(1, len(tr) + 1), tr, label="Train")
            plt.plot(range(1, len(val) + 1), val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} – Accuracy (train vs val)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_accuracy_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error accuracy curve {dname}: {e}")
        plt.close()

    # 3) Confusion matrix
    try:
        if y_true.size and y_pred.size:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.title(f"{dname} – Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.colorbar(im)
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error confusion matrix {dname}: {e}")
        plt.close()

    # ----- collect for aggregate comparison -----
    if metrics.get("val"):
        final_accs.append(metrics["val"][-1])
    else:
        final_accs.append(np.nan)
    irfs.append(dd.get("metrics", {}).get("IRF", [np.nan])[-1])
    dnames.append(dname)

# 4) Aggregate comparison plot (final val acc & IRF)
try:
    if dnames:
        x = np.arange(len(dnames))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, final_accs, width, label="Final Val Acc")
        plt.bar(x + width / 2, irfs, width, label="IRF")
        plt.xticks(x, dnames, rotation=15)
        plt.ylim(0, 1)
        plt.title("Dataset Comparison – Final Validation Accuracy vs IRF")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "datasets_comparison_acc_irf.png"))
        plt.close()
except Exception as e:
    print(f"Error aggregate comparison plot: {e}")
    plt.close()

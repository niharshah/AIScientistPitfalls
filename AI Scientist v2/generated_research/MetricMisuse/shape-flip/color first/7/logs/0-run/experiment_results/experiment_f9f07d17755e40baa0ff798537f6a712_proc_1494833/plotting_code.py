import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# ---------- helpers ----------
def count_color_variety(seq):
    return len({tok[1:] for tok in seq.split()})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def comp_weight(seq):
    return count_color_variety(seq) + count_shape_variety(seq)


def confusion_matrix(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    return cm


# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = list(experiment_data.keys())
val_acc_all = {}

for dname in datasets:
    ds = experiment_data[dname]
    epochs = np.arange(1, len(ds["losses"]["train"]) + 1)

    tr_loss = ds["losses"]["train"]
    val_loss = ds["losses"]["val"]
    tr_acc = [m["acc"] for m in ds["metrics"]["train"]]
    val_acc = [m["acc"] for m in ds["metrics"]["val"]]
    val_cwa = [m["CompWA"] for m in ds["metrics"]["val"]]
    val_acc_all[dname] = val_acc

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot ({dname}): {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dname}: Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot ({dname}): {e}")
        plt.close()

    # 3) CompWA curve
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, label="Validation CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"{dname}: Validation Complexity-Weighted Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_compwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot ({dname}): {e}")
        plt.close()

    # 4) Confusion matrix (test)
    try:
        preds = np.array(ds["predictions"])
        gts = np.array(ds["ground_truth"])
        nc = len(set(gts.tolist() + preds.tolist()))
        cm = confusion_matrix(preds, gts, nc)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        tick = np.arange(nc)
        plt.xticks(tick)
        plt.yticks(tick)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dname}: Confusion Matrix")
        for i, j in itertools.product(range(nc), range(nc)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix ({dname}): {e}")
        plt.close()

    # ---------- evaluation metrics ----------
    try:
        weights = np.array([comp_weight(s) for s in ds["sequences"]])
        test_acc = (preds == gts).mean()
        cowa = (weights * (preds == gts)).sum() / weights.sum()
        print(f"{dname} -- Test Accuracy: {test_acc:.3f} | Test CompWA: {cowa:.3f}")
    except Exception as e:
        print(f"Error computing evaluation metrics ({dname}): {e}")

# 5) Cross-dataset comparison (validation accuracy)
if len(datasets) > 1:
    try:
        plt.figure()
        for dname, vacc in val_acc_all.items():
            plt.plot(np.arange(1, len(vacc) + 1), vacc, label=dname)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()

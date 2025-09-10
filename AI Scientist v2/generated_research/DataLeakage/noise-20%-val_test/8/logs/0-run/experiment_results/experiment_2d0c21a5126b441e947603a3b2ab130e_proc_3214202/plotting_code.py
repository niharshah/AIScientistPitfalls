import matplotlib.pyplot as plt
import numpy as np
import os

# prepare output dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data -------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch the only dataset name stored inside each experiment
def first_ds_key(exp_dict):
    return next(iter(exp_dict.keys()))


# 1) accuracy comparison -------------------------------------------------------
try:
    exp_names = list(experiment_data.keys())
    train_accs = []
    val_accs = []
    test_accs = []

    for exp in exp_names:
        ds_key = first_ds_key(experiment_data[exp])
        res = experiment_data[exp][ds_key]
        train_accs.append(res["metrics"]["train"][0])
        val_accs.append(res["metrics"]["val"][0])
        test_accs.append(res["metrics"]["test"][0])

    x = np.arange(len(exp_names))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, train_accs, width, label="Train")
    plt.bar(x, val_accs, width, label="Val")
    plt.bar(x + width, test_accs, width, label="Test")
    plt.xticks(x, exp_names, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"{ds_key}: Train/Val/Test Accuracy Comparison")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_key}_accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy comparison plot: {e}")
    plt.close()

# 2-4) confusion matrices (limit to 3 experiments incl. baseline) -------------
max_conf = 3
for idx, exp in enumerate(list(experiment_data.keys())[:max_conf]):
    try:
        ds_key = first_ds_key(experiment_data[exp])
        res = experiment_data[exp][ds_key]
        y_true = np.array(res["ground_truth"])
        y_pred = np.array(res["predictions"])

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_key} Confusion Matrix\nExp: {exp}")
        plt.colorbar(im, fraction=0.046)
        fname = os.path.join(working_dir, f"{ds_key}_{exp}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {exp}: {e}")
        plt.close()

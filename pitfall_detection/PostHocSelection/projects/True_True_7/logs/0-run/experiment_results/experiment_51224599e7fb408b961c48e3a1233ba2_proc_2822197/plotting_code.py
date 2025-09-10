import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is None:
    raise SystemExit("No experiment_data.npy found – nothing to plot.")


# helper to count correct/incorrect for confusion matrix
def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m, labels


datasets = list(experiment_data.keys())
test_scores = {}

for ds in datasets:
    data = experiment_data[ds]
    losses = data.get("losses", {})
    metrics = data.get("metrics", {})
    # ---------- 1) loss curves ----------
    try:
        plt.figure()
        x = np.arange(len(losses.get("train", [])))
        if len(x):
            plt.plot(x, losses["train"], ls="--", label="train")
            plt.plot(x, losses["val"], ls="-", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds} Loss Curves\nTrain (dashed) vs Validation (solid)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds.lower()}_loss_curves.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

    # ---------- 2) metric curves ----------
    try:
        plt.figure()
        x = np.arange(len(metrics.get("train", [])))
        if len(x):
            plt.plot(x, metrics["train"], ls="--", label="train SWA")
            plt.plot(x, metrics["val"], ls="-", label="val SWA")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{ds} SWA Curves\nTrain (dashed) vs Validation (solid)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds.lower()}_swa_curves.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {ds}: {e}")
        plt.close()

    # ---------- 3) confusion matrix ----------
    try:
        y_true = data.get("ground_truth", [])
        y_pred = data.get("predictions", [])
        if y_true and y_pred:
            labels = sorted(set(y_true) | set(y_pred))
            mat, lbls = confusion(y_true, y_pred, labels)
            plt.figure()
            plt.imshow(mat, cmap="Blues")
            plt.colorbar()
            plt.xticks(ticks=np.arange(len(lbls)), labels=lbls, rotation=45)
            plt.yticks(ticks=np.arange(len(lbls)), labels=lbls)
            plt.title(f"{ds} Confusion Matrix\nLeft: Ground Truth, Bottom: Predictions")
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    plt.text(j, i, mat[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{ds.lower()}_confusion.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds}: {e}")
        plt.close()

    # store test score
    test_scores[ds] = metrics.get("test", None)

# ---------- 4) bar chart comparing datasets ----------
try:
    plt.figure()
    names, vals = [], []
    for k, v in test_scores.items():
        if v is not None:
            names.append(k)
            vals.append(v)
    if names:
        plt.bar(names, vals, color="skyblue")
        plt.ylabel("Test Shape-Weighted Accuracy")
        plt.title("Final Test SWA by Dataset")
        fname = os.path.join(working_dir, "all_datasets_test_swa.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated bar chart: {e}")
    plt.close()

# ---------- 5) print numerical results ----------
for ds, score in test_scores.items():
    print(f"{ds} Test SWA: {score}")

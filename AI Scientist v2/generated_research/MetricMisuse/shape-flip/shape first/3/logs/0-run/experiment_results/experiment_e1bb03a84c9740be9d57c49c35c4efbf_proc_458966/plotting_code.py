import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

# ----------------- paths & loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# container for comparison bar chart
comp_metrics = {}

for dname, data in experiment_data.items():
    # ----------------- Loss curves -----------------
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], "--", label="train")
        plt.plot(data["losses"]["val"], "-", label="val")
        plt.title(f"{dname} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: error plotting loss curves: {e}")
        plt.close()

    # ----------------- Validation metric curves (assume SWA) -----------------
    try:
        val_metric = data["metrics"]["val"]
        plt.figure()
        plt.plot(val_metric, color="tab:orange")
        plt.title(f"{dname} Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_val_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: error plotting validation metric: {e}")
        plt.close()

    # ----------------- Confusion matrix -----------------
    try:
        y_true, y_pred = data["ground_truth"], data["predictions"]
        labels = sorted(set(y_true))
        idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[idx[t], idx[p]] += 1
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"{dname} Confusion Matrix\nLeft: True, Top: Pred")
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.yticks(range(len(labels)), labels, fontsize=6)
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=6,
            )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: error plotting confusion matrix: {e}")
        plt.close()

    # collect final test metrics for comparison plot
    try:
        comp_metrics[dname] = data["test_metrics"]
        print(
            f"{dname} | loss={data['test_metrics']['loss']:.4f} | "
            f"SWA={data['test_metrics']['SWA']:.4f}"
        )
    except Exception as e:
        print(f"{dname}: error reading test metrics: {e}")

# ----------------- Comparison bar chart across datasets -----------------
try:
    if comp_metrics:
        names = list(comp_metrics.keys())
        swa_vals = [comp_metrics[n]["SWA"] for n in names]
        x = np.arange(len(names))
        plt.figure()
        plt.bar(x, swa_vals, color="tab:green")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel("SWA")
        plt.title("Final Test SWA Comparison Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "datasets_SWA_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error plotting comparison bar chart: {e}")
    plt.close()

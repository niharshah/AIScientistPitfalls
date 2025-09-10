import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------ paths / data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

models = list(data.keys())


# helper to fetch lists quickly
def get_list(m, key, split):
    return [
        d[key]
        for d in data[m][split][split_name]
        for split_name in ("losses", "metrics")
        if split_name == split
    ]


# ------------------------------------------------------------------ Plot 1: Loss curves
try:
    plt.figure()
    for m in models:
        tr = [d["loss"] for d in data[m]["losses"]["train"]]
        vl = [d["loss"] for d in data[m]["losses"]["val"]]
        epochs = [d["epoch"] for d in data[m]["losses"]["train"]]
        plt.plot(epochs, tr, "--", label=f"{m} Train")
        plt.plot(epochs, vl, "-", label=f"{m} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------ Plot 2: Macro-F1 curves
try:
    plt.figure()
    for m in models:
        tr = [d["macro_f1"] for d in data[m]["metrics"]["train"]]
        vl = [d["macro_f1"] for d in data[m]["metrics"]["val"]]
        epochs = [d["epoch"] for d in data[m]["metrics"]["train"]]
        plt.plot(epochs, tr, "--", label=f"{m} Train")
        plt.plot(epochs, vl, "-", label=f"{m} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ------------------------------------------------------------------ Plot 3: Final validation accuracy bar chart
try:
    plt.figure()
    accs = [data[m]["metrics"]["val"][-1]["RGA"] for m in models]
    plt.bar(models, accs, color="skyblue")
    plt.ylabel("Accuracy (RGA)")
    plt.title("SPR_BENCH Final Validation Accuracy")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ Plot 4: Confusion matrix for best model
try:
    # select model with best last-epoch macro-F1
    best = max(models, key=lambda m: data[m]["metrics"]["val"][-1]["macro_f1"])
    y_pred = data[best]["predictions"]
    y_true = data[best]["ground_truth"]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"SPR_BENCH Confusion Matrix – {best}")
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_{best}_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

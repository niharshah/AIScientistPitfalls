import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ---- paths / loading ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
exp_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

models = list(experiment_data.keys())


# Helper to grab lists
def grab(path):
    out = []
    for d in path:
        out.append(d[list(d.keys() - {"epoch"})[0]])  # pick the sole metric/loss key
    return out


# ---- Plot 1: Loss curves ----
try:
    plt.figure()
    for m in models:
        exp = experiment_data[m]["SPR_BENCH"]
        ep = [d["epoch"] for d in exp["losses"]["train"]]
        plt.plot(ep, grab(exp["losses"]["train"]), label=f"{m} Train")
        plt.plot(ep, grab(exp["losses"]["val"]), label=f"{m} Val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---- Plot 2: Macro-F1 curves ----
try:
    plt.figure()
    for m in models:
        exp = experiment_data[m]["SPR_BENCH"]
        ep = [d["epoch"] for d in exp["metrics"]["val"]]
        plt.plot(ep, [d["macro_f1"] for d in exp["metrics"]["val"]], label=m)
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Validation Macro-F1 over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ---- Plot 3: Final validation accuracy bar chart ----
try:
    plt.figure()
    accs = [
        experiment_data[m]["SPR_BENCH"]["metrics"]["val"][-1]["RGA"] for m in models
    ]
    plt.bar(models, accs)
    plt.ylabel("Validation Accuracy (RGA)")
    plt.title("SPR_BENCH: Final Validation Accuracy Comparison")
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---- Plot 4: Confusion matrix for Baseline ----
try:
    base = experiment_data["Baseline"]["SPR_BENCH"]
    y_true, y_pred = base["ground_truth"], base["predictions"]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR_BENCH: Baseline Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fname = os.path.join(working_dir, "SPR_BENCH_baseline_confusion.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---- Print final summary ----
print("Final Validation Metrics:")
for m in models:
    val = experiment_data[m]["SPR_BENCH"]["metrics"]["val"][-1]
    print(f"{m:20s}  Macro-F1: {val['macro_f1']:.3f}  RGA: {val['RGA']:.3f}")

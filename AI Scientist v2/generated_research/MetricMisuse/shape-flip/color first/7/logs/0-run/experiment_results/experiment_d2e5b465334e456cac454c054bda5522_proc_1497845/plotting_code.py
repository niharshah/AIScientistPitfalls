import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# paths & data loading
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_BENCH"
exps = ["FULL", "SEQ_ONLY"]


# ------------------------------------------------------------------
# helper to fetch curves
# ------------------------------------------------------------------
def curve(exp, split, field):
    """Return y values for a given curve (losses or acc)"""
    if exp not in experiment_data:
        return []
    node = experiment_data[exp][dataset_name]
    if field == "loss":
        return node["losses"][split]
    if field == "acc":
        return [m["acc"] for m in node["metrics"][split]]
    return []


# ------------------------------------------------------------------
# 1. Loss curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for exp in exps:
        y_tr = curve(exp, "train", "loss")
        y_val = curve(exp, "val", "loss")
        x = range(1, len(y_tr) + 1)
        plt.plot(x, y_tr, label=f"{exp}-train")
        plt.plot(x, y_val, "--", label=f"{exp}-val")
    plt.title(f"{dataset_name}: Training vs Validation Loss (FULL vs SEQ_ONLY)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2. Accuracy curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for exp in exps:
        y_tr = curve(exp, "train", "acc")
        y_val = curve(exp, "val", "acc")
        x = range(1, len(y_tr) + 1)
        plt.plot(x, y_tr, label=f"{exp}-train")
        plt.plot(x, y_val, "--", label=f"{exp}-val")
    plt.title(f"{dataset_name}: Training vs Validation Accuracy (FULL vs SEQ_ONLY)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3. Bar chart – test ACC
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(4, 4))
    accs = [experiment_data[e][dataset_name]["metrics"]["test"]["acc"] for e in exps]
    plt.bar(exps, accs, color=["tab:blue", "tab:orange"])
    plt.ylim(0, 1)
    plt.title(f"{dataset_name}: Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    fname = os.path.join(working_dir, f"{dataset_name}_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4. Bar chart – test CompWA
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(4, 4))
    cwas = [experiment_data[e][dataset_name]["metrics"]["test"]["CompWA"] for e in exps]
    plt.bar(exps, cwas, color=["tab:green", "tab:red"])
    plt.ylim(0, 1)
    plt.title(f"{dataset_name}: Test Complexity-Weighted Accuracy")
    plt.ylabel("CompWA")
    fname = os.path.join(working_dir, f"{dataset_name}_test_compwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test CompWA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# textual summary
# ------------------------------------------------------------------
for e in exps:
    m = (
        experiment_data.get(e, {})
        .get(dataset_name, {})
        .get("metrics", {})
        .get("test", {})
    )
    if m:
        print(f"{e}  --  ACC: {m['acc']:.3f}   CompWA: {m['CompWA']:.3f}")

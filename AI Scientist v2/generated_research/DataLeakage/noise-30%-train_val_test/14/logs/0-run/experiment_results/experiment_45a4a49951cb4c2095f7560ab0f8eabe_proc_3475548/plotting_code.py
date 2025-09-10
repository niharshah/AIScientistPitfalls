import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------- paths / load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_data = {}

ds_name = "SPR_BENCH"
if ds_name not in exp_data:
    print(f"{ds_name} not found in experiment data")
    exit()

dset = exp_data[ds_name]
models = list(dset.keys())

# ------------------------------------------------- figure 1 : loss curves
try:
    plt.figure()
    for m in models:
        ep = [d["epoch"] for d in dset[m]["losses"]["train"]]
        tr = [d["loss"] for d in dset[m]["losses"]["train"]]
        val = [d["loss"] for d in dset[m]["losses"]["val"]]
        plt.plot(ep, tr, label=f"{m}-train")
        plt.plot(ep, val, "--", label=f"{m}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{ds_name} Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------- figure 2 : macro-F1 curves
try:
    plt.figure()
    for m in models:
        ep = [d["epoch"] for d in dset[m]["metrics"]["train"]]
        tr = [d["macro_f1"] for d in dset[m]["metrics"]["train"]]
        val = [d["macro_f1"] for d in dset[m]["metrics"]["val"]]
        plt.plot(ep, tr, label=f"{m}-train")
        plt.plot(ep, val, "--", label=f"{m}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{ds_name} Macro-F1 Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_macroF1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 plot: {e}")
    plt.close()

# ------------------------------------------------- figure 3 : accuracy curves (RGA in exp dict)
try:
    plt.figure()
    for m in models:
        ep = [d["epoch"] for d in dset[m]["metrics"]["val"]]
        acc = [d["RGA"] for d in dset[m]["metrics"]["val"]]
        plt.plot(ep, acc, label=m)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (RGA)")
    plt.title(f"{ds_name} Validation Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------- figure 4 : ground truth vs baseline predictions
try:
    base_preds = dset["Baseline"]["predictions"]
    base_gts = dset["Baseline"]["ground_truth"]
    if base_preds and len(base_preds) == len(base_gts):
        classes = sorted(set(base_gts))
        gt_counts = [base_gts.count(c) for c in classes]
        pr_counts = [base_preds.count(c) for c in classes]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar(classes, gt_counts, color="tab:blue")
        ax[0].set_title("Ground Truth")
        ax[1].bar(classes, pr_counts, color="tab:orange")
        ax[1].set_title("Baseline Predictions")
        fig.suptitle(f"{ds_name} – Left: Ground Truth, Right: Generated Samples")
        for a in ax:
            a.set_xlabel("Class id")
            a.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_GT_vs_Pred_Baseline.png"))
        plt.close()
except Exception as e:
    print(f"Error creating GT vs prediction plot: {e}")
    plt.close()

# ------------------------------------------------- quick metric printout
for m in models:
    last_val = dset[m]["metrics"]["val"][-1]
    print(
        f"{m:25s} | Epoch {last_val['epoch']} | Val Macro-F1={last_val['macro_f1']:.3f} | RGA={last_val['RGA']:.3f}"
    )

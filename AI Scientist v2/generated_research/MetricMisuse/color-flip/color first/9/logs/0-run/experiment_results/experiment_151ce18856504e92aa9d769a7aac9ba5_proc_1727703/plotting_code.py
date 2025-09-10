import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# basic set-up
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment results
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
exp = experiment_data.get(dataset, {})

# ------------------------------------------------------------------
# 1) loss curves ----------------------------------------------------
# ------------------------------------------------------------------
try:
    tr_tuples = exp.get("losses", {}).get("train", [])
    val_tuples = exp.get("losses", {}).get("val", [])
    if tr_tuples or val_tuples:
        x_tr = np.arange(1, len(tr_tuples) + 1)
        y_tr = [v for (_, v) in tr_tuples]
        x_val = np.arange(1, len(val_tuples) + 1)
        y_val = [v for (_, v) in val_tuples]

        plt.figure()
        if x_tr.size:
            plt.plot(x_tr, y_tr, label="Train")
        if x_val.size:
            plt.plot(x_val, y_val, label="Validation")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{dataset}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) validation weighted-accuracy curves ---------------------------
# ------------------------------------------------------------------
try:
    val_metrics = exp.get("metrics", {}).get("val", [])
    if val_metrics:
        x = np.arange(1, len(val_metrics) + 1)
        cwa = [m[1] for m in val_metrics]
        swa = [m[2] for m in val_metrics]
        hwa = [m[3] for m in val_metrics]

        plt.figure()
        plt.plot(x, cwa, label="CWA")
        plt.plot(x, swa, label="SWA")
        plt.plot(x, hwa, label="HWA")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.title(f"{dataset}: Validation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset}_val_weighted_accuracies.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) label distribution --------------------------------------------
# ------------------------------------------------------------------
try:
    preds = exp.get("predictions", [])
    gts = exp.get("ground_truth", [])
    if preds and gts:
        labels = sorted(set(gts) | set(preds))
        gt_counts = [gts.count(l) for l in labels]
        pr_counts = [preds.count(l) for l in labels]
        idx = np.arange(len(labels))

        plt.figure()
        bar_w = 0.4
        plt.bar(idx - bar_w / 2, gt_counts, bar_w, label="Ground Truth")
        plt.bar(idx + bar_w / 2, pr_counts, bar_w, label="Predictions")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title(
            f"{dataset}: Label Distribution\nLeft: Ground Truth, Right: Predictions"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset}_label_distribution.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating label distribution plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# print final test metrics -----------------------------------------
# ------------------------------------------------------------------
test_metrics = exp.get("metrics", {}).get("test", None)
if test_metrics:
    cwa, swa, hwa = test_metrics
    print(f"Test Results — CWA: {cwa:.3f}, SWA: {swa:.3f}, HWA: {hwa:.3f}")

import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for ds, rec in experiment_data.items():
    # 1) Loss curves ---------------------------------------------------------
    try:
        plt.figure()
        plt.plot(rec["losses"]["train"], label="train")
        plt.plot(rec["losses"]["val"], label="validation", linestyle="--")
        plt.title(f"{ds} Loss Curves\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

    # 2) SWA curves ----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(rec["metrics"]["train_swa"], label="train SWA")
        plt.plot(rec["metrics"]["val_swa"], label="val   SWA", linestyle="--")
        plt.title(f"{ds} Shape-Weighted Accuracy\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {ds}: {e}")
        plt.close()

# 3) Final test SWA comparison across datasets -------------------------------
try:
    names, test_swas = zip(*[(d, r["test_swa"]) for d, r in experiment_data.items()])
    x = np.arange(len(names))
    plt.figure()
    plt.bar(x, test_swas, color="mediumseagreen")
    plt.xticks(x, names, rotation=15)
    plt.title("Final Test Shape-Weighted Accuracy per Dataset")
    plt.ylabel("SWA")
    plt.savefig(os.path.join(working_dir, "all_datasets_test_swa.png"))
    plt.close()
except Exception as e:
    print(f"Error creating global SWA bar chart: {e}")
    plt.close()

# 4) Scatter GT vs Pred (per dataset) ----------------------------------------
for ds, rec in experiment_data.items():
    try:
        gt, pr = rec.get("ground_truth", []), rec.get("predictions", [])
        if len(gt) == 0 or len(gt) != len(pr):
            continue
        # subsample for readability
        idx = np.random.choice(len(gt), size=min(3000, len(gt)), replace=False)
        plt.figure()
        plt.scatter(np.array(gt)[idx], np.array(pr)[idx], alpha=0.3, s=8)
        plt.title(f"{ds} Ground Truth vs Predictions (scatter)")
        plt.xlabel("Ground Truth label id")
        plt.ylabel("Predicted label id")
        plt.savefig(os.path.join(working_dir, f"{ds}_gt_vs_pred_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter for {ds}: {e}")
        plt.close()

# 5) Histogram of errors -----------------------------------------------------
for ds, rec in experiment_data.items():
    try:
        gt, pr = rec.get("ground_truth", []), rec.get("predictions", [])
        if len(gt) == 0 or len(gt) != len(pr):
            continue
        err_ids = [g for g, p in zip(gt, pr) if g != p]
        if not err_ids:
            continue
        plt.figure()
        plt.hist(err_ids, bins=len(set(gt)), color="coral")
        plt.title(f"{ds} Distribution of Misclassified Ground-Truth Labels")
        plt.xlabel("Label id")
        plt.ylabel("# Errors")
        plt.savefig(os.path.join(working_dir, f"{ds}_error_hist.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating error histogram for {ds}: {e}")
        plt.close()

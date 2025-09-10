import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ---- basic setup -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# paths provided by prompt
experiment_data_path_list = [
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_8e17faf107614e4d9be882114528ec16_proc_3096998/experiment_data.npy",
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_789e750a7306425194fa2d83fef49800_proc_3097000/experiment_data.npy",
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_b5faed0fe34848dc9afb65919550e1cc_proc_3097001/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiment_data:
    print("No experiment data loaded.")
    exit()


# ---- utilities -------------------------------------------------------------
def compute_mean_se(list_of_lists):
    """Stack 1-D lists to 2-D array, truncate to shortest length, return mean & SE."""
    min_len = min(len(l) for l in list_of_lists)
    if min_len == 0:
        return None, None
    arr = np.array([l[:min_len] for l in list_of_lists], dtype=float)
    mean = arr.mean(axis=0)
    se = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, se


# assume every run uses the same dataset key(s); grab intersection
dataset_keys = set.intersection(*(set(d.keys()) for d in all_experiment_data))
if not dataset_keys:
    print("No common dataset key across runs.")
    exit()

# ---------------------------------------------------------------------------
for dname in dataset_keys:
    # gather per-run arrays ---------------------------------------------------
    train_losses_runs, val_losses_runs, val_metric_runs = [], [], []
    preds_runs, gts_runs = [], []
    for run in all_experiment_data:
        data = run[dname]
        train_losses_runs.append(data.get("losses", {}).get("train", []))
        val_losses_runs.append(data.get("losses", {}).get("val", []))
        if "val" in data.get("metrics", {}):
            val_metric_runs.append(data["metrics"]["val"])
        preds_runs.append(data.get("predictions", []))
        gts_runs.append(data.get("ground_truth", []))

    # ---- plot 1: aggregated loss curves ------------------------------------
    try:
        mean_train, se_train = compute_mean_se(train_losses_runs)
        mean_val, se_val = compute_mean_se(val_losses_runs)
        if mean_train is not None or mean_val is not None:
            plt.figure()
            if mean_train is not None:
                x = np.arange(1, len(mean_train) + 1)
                plt.plot(x, mean_train, label="Train Mean")
                plt.fill_between(
                    x,
                    mean_train - se_train,
                    mean_train + se_train,
                    alpha=0.3,
                    label="Train ± SE",
                )
            if mean_val is not None:
                x = np.arange(1, len(mean_val) + 1)
                plt.plot(x, mean_val, label="Val Mean")
                plt.fill_between(
                    x, mean_val - se_val, mean_val + se_val, alpha=0.3, label="Val ± SE"
                )
            plt.title(
                f"{dname}: Aggregated Loss Curves\n(Mean ± Standard Error across {len(all_experiment_data)} runs)"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_agg_loss_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # ---- plot 2: aggregated validation metric ------------------------------
    try:
        if val_metric_runs and all(len(v) for v in val_metric_runs):
            mean_m, se_m = compute_mean_se(val_metric_runs)
            if mean_m is not None:
                x = np.arange(1, len(mean_m) + 1)
                plt.figure()
                plt.plot(x, mean_m, marker="o", label="Val Metric Mean")
                plt.fill_between(
                    x, mean_m - se_m, mean_m + se_m, alpha=0.3, label="Val ± SE"
                )
                plt.title(
                    f"{dname}: Aggregated Validation Metric\n(Mean ± SE across {len(all_experiment_data)} runs)"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Metric value")
                plt.legend()
                fname = os.path.join(working_dir, f"{dname}_agg_val_metric.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric plot for {dname}: {e}")
        plt.close()

    # ---- plot 3: aggregated class distribution -----------------------------
    try:
        # Build union of class ids over all runs
        classes = sorted(set().union(*gts_runs, *preds_runs))
        if classes:
            gt_counts_runs = []
            pred_counts_runs = []
            for gt, pr in zip(gts_runs, preds_runs):
                gt_counts_runs.append([gt.count(c) for c in classes])
                pred_counts_runs.append([pr.count(c) for c in classes])

            gt_mean = np.mean(gt_counts_runs, axis=0)
            pred_mean = np.mean(pred_counts_runs, axis=0)
            gt_se = (
                np.std(gt_counts_runs, ddof=1, axis=0) / sqrt(len(gt_counts_runs))
                if len(gt_counts_runs) > 1
                else np.zeros_like(gt_mean)
            )
            pred_se = (
                np.std(pred_counts_runs, ddof=1, axis=0) / sqrt(len(pred_counts_runs))
                if len(pred_counts_runs) > 1
                else np.zeros_like(pred_mean)
            )

            x = np.arange(len(classes))
            width = 0.35
            plt.figure(figsize=(max(6, len(classes) * 0.8), 4.5))
            plt.bar(
                x - width / 2,
                gt_mean,
                width,
                yerr=gt_se,
                label="Ground Truth",
                capsize=3,
            )
            plt.bar(
                x + width / 2,
                pred_mean,
                width,
                yerr=pred_se,
                label="Predictions",
                capsize=3,
            )
            plt.xticks(x, classes, rotation=45, ha="right")
            plt.title(f"{dname}: Aggregated Class Distribution (Mean ± SE)")
            plt.xlabel("Class ID")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dname}_agg_class_distribution.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated class distribution for {dname}: {e}")
        plt.close()

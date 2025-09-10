import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths / load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# paths provided by the task
experiment_data_path_list = [
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_124a764454534de6b80705c83a7bb620_proc_332229/experiment_data.npy",
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_0bf873255d0a49ab8abfb2c6f6f9c705_proc_332228/experiment_data.npy",
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_f6dc7289184d445c954cd134b202f6ae_proc_332230/experiment_data.npy",
]

# load every run
all_runs = []
try:
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_p, allow_pickle=True).item()
        all_runs.append(exp)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# group by dataset_key
datasets = {}
for run in all_runs:
    for dk, log in run.get("epochs_tuning", {}).items():
        datasets.setdefault(dk, []).append(log)


# ---- helper for confusion matrix
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------------- plotting aggregated results
for dataset_key, logs in datasets.items():
    n_runs = len(logs)
    if n_runs == 0:
        continue

    # Align epochs by the minimum number available across runs
    min_len = min(len(l["epochs"]) for l in logs)
    epochs = np.asarray(logs[0]["epochs"][:min_len])

    # Gather curves
    losses_train = np.stack([np.asarray(l["losses"]["train"][:min_len]) for l in logs])
    losses_dev = np.stack([np.asarray(l["losses"]["dev"][:min_len]) for l in logs])
    pha_train = np.stack(
        [np.asarray(l["metrics"]["train_PHA"][:min_len]) for l in logs]
    )
    pha_dev = np.stack([np.asarray(l["metrics"]["dev_PHA"][:min_len]) for l in logs])

    def mean_stderr(arr):
        mean = arr.mean(axis=0)
        stderr = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, stderr

    # 1) Aggregated Loss curves -------------------------------------------------
    try:
        plt.figure()
        for arr, lbl, col in [
            (losses_train, "train", "tab:blue"),
            (losses_dev, "dev", "tab:orange"),
        ]:
            m, se = mean_stderr(arr)
            plt.plot(epochs, m, color=col, label=f"{lbl} mean")
            plt.fill_between(
                epochs, m - se, m + se, color=col, alpha=0.3, label=f"{lbl} ± stderr"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_key} Aggregated Loss Curve (N={n_runs})")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_agg_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dataset_key}: {e}")
        plt.close()

    # 2) Aggregated PHA curves --------------------------------------------------
    try:
        plt.figure()
        for arr, lbl, col in [
            (pha_train, "train_PHA", "tab:green"),
            (pha_dev, "dev_PHA", "tab:red"),
        ]:
            m, se = mean_stderr(arr)
            plt.plot(epochs, m, color=col, label=f"{lbl} mean")
            plt.fill_between(
                epochs, m - se, m + se, color=col, alpha=0.3, label=f"{lbl} ± stderr"
            )
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title(f"{dataset_key} Aggregated PHA Curve (N={n_runs})")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_agg_pha_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated PHA curve for {dataset_key}: {e}")
        plt.close()

    # 3) Aggregated Test Metrics -----------------------------------------------
    try:
        # collect metric names
        metric_names = list(logs[0].get("test_metrics", {}).keys())
        if metric_names:
            metric_vals = np.array(
                [
                    [log["test_metrics"].get(m, np.nan) for m in metric_names]
                    for log in logs
                ]
            )
            means = np.nanmean(metric_vals, axis=0)
            stderrs = np.nanstd(metric_vals, axis=0, ddof=1) / np.sqrt(
                metric_vals.shape[0]
            )

            x = np.arange(len(metric_names))
            plt.figure()
            plt.bar(x, means, yerr=stderrs, capsize=5, color="tab:purple")
            plt.ylim(0, 1)
            plt.xticks(x, metric_names)
            plt.title(f"{dataset_key} Aggregated Test Metrics (mean ± stderr)")
            for i, v in enumerate(means):
                plt.text(i, v + 0.03, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, f"{dataset_key}_agg_test_metrics.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metric bar plot for {dataset_key}: {e}")
        plt.close()

    # 4) Aggregated Confusion Matrix -------------------------------------------
    try:
        # check if every run has ground truth and predictions
        y_trues, y_preds = [], []
        for l in logs:
            if "ground_truth" in l and "predictions" in l:
                y_trues.append(np.asarray(l["ground_truth"]))
                y_preds.append(np.asarray(l["predictions"]))

        if y_trues and all(len(t) == len(y_trues[0]) for t in y_trues):
            y_true_concat = np.concatenate(y_trues)
            y_pred_concat = np.concatenate(y_preds)
            n_classes = max(y_true_concat.max(), y_pred_concat.max()) + 1
            cm = confusion_matrix(y_true_concat, y_pred_concat, n_classes)

            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dataset_key} Aggregated Confusion Matrix")
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            fname = os.path.join(working_dir, f"{dataset_key}_agg_confusion_matrix.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dataset_key}: {e}")
        plt.close()

print("Aggregated plotting complete; figures saved to", working_dir)

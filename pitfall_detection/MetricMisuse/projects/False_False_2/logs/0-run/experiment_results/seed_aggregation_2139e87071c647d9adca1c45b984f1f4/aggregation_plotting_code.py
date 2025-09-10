import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths / load data -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_4b272fb792324beb9268efc4532400b5_proc_329318/experiment_data.npy",
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_ab9d5da2b5d445968e1366e0c5631190_proc_329320/experiment_data.npy",
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_48ce2daaf8ab4012ab85dc11fe2cad73_proc_329321/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        d = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(d)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded – nothing to plot.")
    exit()


# ---------------- helpers -----------------------------------------------------------
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def aggregate_metric(list_of_runs, key_chain):
    """Return list of np.arrays (one per run) extracted by the key_chain."""
    arrays = []
    for run in list_of_runs:
        try:
            arr = run
            for k in key_chain:
                arr = arr[k]
            arrays.append(np.asarray(arr))
        except KeyError:
            continue
    return arrays


# ---------------- aggregation / plotting --------------------------------------------
# assume all runs share the same dataset keys
dataset_keys = set()
for run in all_experiment_data:
    dataset_keys.update(run.get("epochs_tuning", {}).keys())

for dataset in dataset_keys:
    # ---------------------------------------------------------------- Loss curves
    try:
        train_losses = aggregate_metric(
            [
                r["epochs_tuning"][dataset]
                for r in all_experiment_data
                if dataset in r.get("epochs_tuning", {})
            ],
            ["losses", "train"],
        )
        dev_losses = aggregate_metric(
            [
                r["epochs_tuning"][dataset]
                for r in all_experiment_data
                if dataset in r.get("epochs_tuning", {})
            ],
            ["losses", "dev"],
        )
        epochs_list = aggregate_metric(
            [
                r["epochs_tuning"][dataset]
                for r in all_experiment_data
                if dataset in r.get("epochs_tuning", {})
            ],
            ["epochs"],
        )
        if train_losses and epochs_list:
            min_len = min(map(len, train_losses))
            train_stack = np.stack([tl[:min_len] for tl in train_losses])
            dev_stack = (
                np.stack([dl[:min_len] for dl in dev_losses]) if dev_losses else None
            )
            epochs = epochs_list[0][:min_len]

            mean_train = train_stack.mean(0)
            sem_train = train_stack.std(0, ddof=1) / np.sqrt(train_stack.shape[0])

            plt.figure()
            plt.errorbar(
                epochs,
                mean_train,
                yerr=sem_train,
                label="train (mean±SEM)",
                color="tab:blue",
            )

            if dev_stack is not None:
                mean_dev = dev_stack.mean(0)
                sem_dev = dev_stack.std(0, ddof=1) / np.sqrt(dev_stack.shape[0])
                plt.errorbar(
                    epochs,
                    mean_dev,
                    yerr=sem_dev,
                    label="dev (mean±SEM)",
                    color="tab:orange",
                )

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dataset} Loss Curve (Mean ± SEM) – aggregated over {train_stack.shape[0]} runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset}_aggregated_loss_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dataset}: {e}")
        plt.close()

    # ---------------------------------------------------------------- PHA curves
    try:
        train_pha = aggregate_metric(
            [
                r["epochs_tuning"][dataset]
                for r in all_experiment_data
                if dataset in r.get("epochs_tuning", {})
            ],
            ["metrics", "train_PHA"],
        )
        dev_pha = aggregate_metric(
            [
                r["epochs_tuning"][dataset]
                for r in all_experiment_data
                if dataset in r.get("epochs_tuning", {})
            ],
            ["metrics", "dev_PHA"],
        )
        if train_pha:
            min_len = min(map(len, train_pha))
            train_stack = np.stack([tp[:min_len] for tp in train_pha])
            dev_stack = np.stack([dp[:min_len] for dp in dev_pha]) if dev_pha else None
            epochs = epochs_list[0][:min_len]

            mean_train = train_stack.mean(0)
            sem_train = train_stack.std(0, ddof=1) / np.sqrt(train_stack.shape[0])

            plt.figure()
            plt.errorbar(
                epochs,
                mean_train,
                yerr=sem_train,
                label="train_PHA (mean±SEM)",
                color="tab:green",
            )

            if dev_stack is not None:
                mean_dev = dev_stack.mean(0)
                sem_dev = dev_stack.std(0, ddof=1) / np.sqrt(dev_stack.shape[0])
                plt.errorbar(
                    epochs,
                    mean_dev,
                    yerr=sem_dev,
                    label="dev_PHA (mean±SEM)",
                    color="tab:red",
                )

            plt.xlabel("Epoch")
            plt.ylabel("PHA")
            plt.title(f"{dataset} PHA Curve (Mean ± SEM) – aggregated")
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset}_aggregated_pha_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated PHA curve for {dataset}: {e}")
        plt.close()

    # ---------------------------------------------------------------- Test metrics bar
    try:
        test_metric_dicts = [
            r["epochs_tuning"][dataset].get("test_metrics", {})
            for r in all_experiment_data
            if dataset in r.get("epochs_tuning", {})
        ]
        if test_metric_dicts and all(test_metric_dicts):
            metric_names = list(test_metric_dicts[0].keys())
            metric_vals = np.array(
                [[d[m] for m in metric_names] for d in test_metric_dicts]
            )
            means = metric_vals.mean(0)
            sems = metric_vals.std(0, ddof=1) / np.sqrt(metric_vals.shape[0])

            plt.figure()
            x = np.arange(len(metric_names))
            plt.bar(x, means, yerr=sems, capsize=5, color="tab:blue")
            plt.ylim(0, 1)
            plt.xticks(x, metric_names)
            plt.title(
                f"{dataset} Test Metrics (Mean ± SEM) – {metric_vals.shape[0]} runs"
            )
            for i, v in enumerate(means):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, f"{dataset}_aggregated_test_metrics.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metric bar plot for {dataset}: {e}")
        plt.close()

    # ---------------------------------------------------------------- Confusion matrix
    try:
        gt_preds = [
            (
                np.asarray(r["epochs_tuning"][dataset].get("ground_truth", [])),
                np.asarray(r["epochs_tuning"][dataset].get("predictions", [])),
            )
            for r in all_experiment_data
            if dataset in r.get("epochs_tuning", {})
        ]
        gt_preds = [(gt, pr) for gt, pr in gt_preds if gt.size and pr.size]
        if gt_preds:
            n_classes = max(max(gt.max(), pr.max()) for gt, pr in gt_preds) + 1
            agg_cm = np.zeros((n_classes, n_classes), dtype=int)
            for gt, pr in gt_preds:
                agg_cm += confusion_matrix(gt, pr, n_classes)

            plt.figure()
            plt.imshow(agg_cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dataset} Confusion Matrix – aggregated over {len(gt_preds)} runs"
            )
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(
                        j,
                        i,
                        agg_cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if agg_cm[i, j] > agg_cm.max() / 2 else "black",
                    )
            fname = os.path.join(
                working_dir, f"{dataset}_aggregated_confusion_matrix.png"
            )
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dataset}: {e}")
        plt.close()

print("Aggregated plotting complete; figures saved to", working_dir)

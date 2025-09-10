import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment paths ----------
experiment_data_path_list = [
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_52f6f112e5604cd29c71914bbd857e12_proc_1509393/experiment_data.npy",
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_a0b9d35d19bb410ebc2c7e2f777176e7_proc_1509394/experiment_data.npy",
    "experiments/2025-08-30_21-49-50_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_41c91bd2f57b40f1a882124ecea2d722_proc_1509392/experiment_data.npy",
]

# ---------- load all experiments ----------
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ---------- utility ----------
def stack_and_aggr(list_of_lists):
    """Stack 1-D arrays from different runs after truncating to the minimum length."""
    if not list_of_lists:
        return None, None, None
    min_len = min(len(a) for a in list_of_lists)
    arr = np.stack(
        [np.asarray(a[:min_len]) for a in list_of_lists], axis=0
    )  # (runs, T)
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem, np.arange(min_len)


# ---------- aggregate across datasets ----------
dataset_names = set()
for exp in all_experiment_data:
    dataset_names.update(exp.keys())

for ds_name in dataset_names:
    # gather per-run arrays
    train_loss_runs, val_loss_runs = [], []
    train_acc_runs, val_acc_runs = [], []
    preds_runs, gts_runs = [], []

    for exp in all_experiment_data:
        ds_content = exp.get(ds_name, {})
        losses = ds_content.get("losses", {})
        metrics = ds_content.get("metrics", {})
        if "train" in losses and len(losses["train"]):
            train_loss_runs.append(losses["train"])
        if "val" in losses and len(losses["val"]):
            val_loss_runs.append(losses["val"])
        if "train" in metrics and len(metrics["train"]):
            train_acc_runs.append(metrics["train"])
        if "val" in metrics and len(metrics["val"]):
            val_acc_runs.append(metrics["val"])
        preds = ds_content.get("predictions")
        gts = ds_content.get("ground_truth")
        if preds is not None and gts is not None and len(preds) and len(gts):
            preds_runs.append(np.asarray(preds))
            gts_runs.append(np.asarray(gts))

    # ========== aggregated LOSS curve ==========
    try:
        mean_tr, sem_tr, x = stack_and_aggr(train_loss_runs)
        mean_val, sem_val, _ = stack_and_aggr(val_loss_runs)
        if mean_tr is not None or mean_val is not None:
            plt.figure()
            if mean_tr is not None:
                plt.plot(x, mean_tr, label="Train mean")
                plt.fill_between(
                    x,
                    mean_tr - sem_tr,
                    mean_tr + sem_tr,
                    alpha=0.3,
                    label="Train ± SEM",
                )
            if mean_val is not None:
                plt.plot(x, mean_val, label="Val mean")
                plt.fill_between(
                    x,
                    mean_val - sem_val,
                    mean_val + sem_val,
                    alpha=0.3,
                    label="Val ± SEM",
                )
            plt.title(f"{ds_name} Aggregated Loss Curve\nShaded: ± Standard Error")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = f"{ds_name}_aggregated_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss for {ds_name}: {e}")
        plt.close()

    # ========== aggregated ACCURACY curve ==========
    try:
        mean_tr, sem_tr, x = stack_and_aggr(train_acc_runs)
        mean_val, sem_val, _ = stack_and_aggr(val_acc_runs)
        if mean_tr is not None or mean_val is not None:
            plt.figure()
            if mean_tr is not None:
                plt.plot(x, mean_tr, label="Train mean")
                plt.fill_between(
                    x,
                    mean_tr - sem_tr,
                    mean_tr + sem_tr,
                    alpha=0.3,
                    label="Train ± SEM",
                )
            if mean_val is not None:
                plt.plot(x, mean_val, label="Val mean")
                plt.fill_between(
                    x,
                    mean_val - sem_val,
                    mean_val + sem_val,
                    alpha=0.3,
                    label="Val ± SEM",
                )
            plt.title(f"{ds_name} Aggregated Accuracy Curve\nShaded: ± Standard Error")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = f"{ds_name}_aggregated_accuracy_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy for {ds_name}: {e}")
        plt.close()

    # ========== aggregated CONFUSION matrix ==========
    try:
        if preds_runs and gts_runs and len(preds_runs) == len(gts_runs):
            # compute summed confusion matrix
            num_classes = int(
                max(
                    max(pr.max() for pr in preds_runs), max(gt.max() for gt in gts_runs)
                )
                + 1
            )
            agg_cm = np.zeros((num_classes, num_classes), dtype=int)
            for pr, gt in zip(preds_runs, gts_runs):
                for g, p in zip(gt, pr):
                    agg_cm[g, p] += 1
            plt.figure()
            im = plt.imshow(agg_cm, cmap="Blues")
            plt.colorbar(im)
            ticks = np.arange(num_classes)
            plt.xticks(ticks, [f"c{i}" for i in ticks])
            plt.yticks(ticks, [f"c{i}" for i in ticks])
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{ds_name} Aggregated Confusion Matrix\nCounts across runs")
            fname = f"{ds_name}_aggregated_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        else:
            print(f"No confusion matrix data for {ds_name}")
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {ds_name}: {e}")
        plt.close()

    # ---------- print summary metric ----------
    if val_acc_runs:
        last_vals = [arr[-1] for arr in val_acc_runs if len(arr)]
        mean_last = np.mean(last_vals)
        sem_last = np.std(last_vals, ddof=1) / np.sqrt(len(last_vals))
        print(f"{ds_name} final validation accuracy: {mean_last:.4f} ± {sem_last:.4f}")

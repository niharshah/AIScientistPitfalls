import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list of experiment data files ----------
experiment_data_path_list = [
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_3c25ab3e62654a9e8ca9b51477e9583f_proc_3066821/experiment_data.npy",
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_1a3b94efc3994970a57aa504f4f4fcab_proc_3066824/experiment_data.npy",
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_c17d8bd5d4cf4cbca8b9d7d41f6f8b50_proc_3066823/experiment_data.npy",
]

# ---------- load all experiment data ----------
all_experiments = []
for pth in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), pth)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiments.append(exp)
    except Exception as e:
        print(f"Error loading {pth}: {e}")


# ---------- helper to aggregate over runs ----------
def aggregate_epoch_metrics(metric_lists):
    """
    metric_lists: list of (N_i,2) arrays from different runs
    returns epochs_sorted, mean_vals, sem_vals (all np.ndarray)
    """
    # collect union of epochs
    epochs = sorted({int(e) for arr in metric_lists for e in arr[:, 0]})
    epoch_to_vals = defaultdict(list)
    for arr in metric_lists:
        for e, v in arr:
            epoch_to_vals[int(e)].append(v)
    mean_vals, sem_vals = [], []
    for e in epochs:
        vals = np.array(epoch_to_vals.get(e, [np.nan]), dtype=float)
        mean_vals.append(np.nanmean(vals))
        sem_vals.append(np.nanstd(vals, ddof=1) / np.sqrt(np.isfinite(vals).sum()))
    return np.array(epochs), np.array(mean_vals), np.array(sem_vals)


# ---------- iterate through every dataset encountered ----------
datasets = set()
for exp in all_experiments:
    datasets.update(exp.keys())

for dset_name in datasets:
    # gather per-run data structures
    train_lists, val_lists, acs_lists = [], [], []
    conf_mats = []
    num_classes = 0

    for exp in all_experiments:
        dset = exp.get(dset_name, {})
        metrics = dset.get("metrics", {})
        preds = np.array(dset.get("predictions", []))
        gts = np.array(dset.get("ground_truth", []))

        # collect scalar epoch metrics if present
        train_arr = np.array(metrics.get("train_loss", []))
        val_arr = np.array(metrics.get("val_loss", []))
        acs_arr = np.array(metrics.get("val_ACS", []))

        if train_arr.size:
            train_lists.append(train_arr)
        if val_arr.size:
            val_lists.append(val_arr)
        if acs_arr.size:
            acs_lists.append(acs_arr)

        # accumulate confusion matrix
        if preds.size and gts.size:
            run_classes = max(max(preds), max(gts)) + 1
            run_cm = np.zeros((run_classes, run_classes), dtype=int)
            for p, g in zip(preds, gts):
                run_cm[g, p] += 1
            conf_mats.append(run_cm)
            num_classes = max(num_classes, run_classes)

    # ---- 1) aggregated train/val loss ----
    try:
        if train_lists and val_lists:
            epochs_t, mean_t, sem_t = aggregate_epoch_metrics(train_lists)
            epochs_v, mean_v, sem_v = aggregate_epoch_metrics(val_lists)

            plt.figure()
            plt.fill_between(epochs_t, mean_t - sem_t, mean_t + sem_t, alpha=0.2)
            plt.plot(epochs_t, mean_t, label="Train Loss (mean)")
            plt.fill_between(epochs_v, mean_v - sem_v, mean_v + sem_v, alpha=0.2)
            plt.plot(epochs_v, mean_v, label="Validation Loss (mean)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dset_name}: Mean ± SEM Loss Curves\n(Left: Ground Truth, Right: Generated Samples)"
            )
            plt.legend()
            fname = f"{dset_name}_agg_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # ---- 2) aggregated validation ACS ----
    try:
        if acs_lists:
            epochs_a, mean_a, sem_a = aggregate_epoch_metrics(acs_lists)
            plt.figure()
            plt.fill_between(
                epochs_a, mean_a - sem_a, mean_a + sem_a, alpha=0.2, label="±1 SEM"
            )
            plt.plot(epochs_a, mean_a, marker="o", label="Mean ACS")
            plt.xlabel("Epoch")
            plt.ylabel("ACS")
            plt.ylim(0, 1)
            plt.title(
                f"{dset_name}: Mean ± SEM Validation ACS\n(Left: Ground Truth, Right: Generated Samples)"
            )
            plt.legend()
            fname = f"{dset_name}_agg_ACS_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated ACS plot for {dset_name}: {e}")
        plt.close()

    # ---- 3) aggregated confusion matrix ----
    try:
        if conf_mats:
            # pad confusion matrices to same size then sum
            padded = []
            for cm in conf_mats:
                pad_cm = np.zeros((num_classes, num_classes), dtype=int)
                pad_cm[: cm.shape[0], : cm.shape[1]] = cm
                padded.append(pad_cm)
            cm_total = np.sum(padded, axis=0)
            cm_percent = cm_total / cm_total.sum(axis=1, keepdims=True)
            plt.figure()
            im = plt.imshow(cm_percent, cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dset_name}: Aggregated Confusion Matrix (%)\n(Left: Ground Truth, Right: Generated Samples)"
            )
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(
                        j,
                        i,
                        f"{cm_percent[i, j]*100:0.1f}%",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=6,
                    )
            fname = f"{dset_name}_agg_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dset_name}: {e}")
        plt.close()

print("Finished saving aggregated plots to", working_dir)
